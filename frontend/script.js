// =========================
// Basic config
// =========================

// host / port
const API_BASE = "http://50.18.67.156:5001";
//const socket = io("http://50.18.67.156:5001"); // move to initSocket function
let socket = null;

let authToken = null;
let currentUser = null;

let isTableHost = false;

let gameFinished = false;
let matchFinished = false;

const MAX_REDRAWS = 7;
let redrawRemaining = 0;
let selectedCards = new Set();

const MAX_ROUNDS = 3;
let playerScore = 0;
let opponentScore = 0;

let playerHand = [];
let opponentHand = [];
let deckPool = [];

let gameMode = null;
let gameId = null;
let player = 1;
let round = 1;
let rId = null;

// Card helpers
const SUITS = ["♣", "♦", "♥", "♠"];
const RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"];

document.addEventListener("DOMContentLoaded", () => {
    setupAuthTabSwitch();
    setupAuthForms();
    setupLobbyButtons();
    setupGameButtons();
});


// =========================
// Auth view
// =========================

function setupAuthTabSwitch() {
    const tabLogin = document.getElementById("tab-login");
    const tabRegister = document.getElementById("tab-register");
    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");

    tabLogin.addEventListener("click", () => {
        tabLogin.classList.add("active");
        tabRegister.classList.remove("active");
        loginForm.classList.remove("hidden");
        registerForm.classList.add("hidden");
    });

    tabRegister.addEventListener("click", () => {
        tabRegister.classList.add("active");
        tabLogin.classList.remove("active");
        registerForm.classList.remove("hidden");
        loginForm.classList.add("hidden");
    });
}

function setupAuthForms() {
    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");

    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await doLogin();
    });

    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await doRegister();
    });
}

async function doRegister() {
    const username = document.getElementById("register-username").value.trim();
    const password = document.getElementById("register-password").value.trim();
    const statusEl = document.getElementById("register-status");
    statusEl.textContent = "";

    if (!username || !password) {
        statusEl.textContent = "Please enter username and password.";
        statusEl.className = "status-message status-error";
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/user/register`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ username, password })
        });

        const data = await res.json();

        if (!res.ok) {
            statusEl.textContent = data.error || "Register failed.";
            statusEl.className = "status-message status-error";
            return;
        }

        statusEl.textContent = "Register success. You can log in now.";
        statusEl.className = "status-message status-ok";

        // Optional: auto-fill login form
        document.getElementById("login-username").value = username;
        document.getElementById("login-password").value = "";

    } catch (err) {
        console.error(err);
        statusEl.textContent = "Network error.";
        statusEl.className = "status-message status-error";
    }
}

async function doLogin() {
    const username = document.getElementById("login-username").value.trim();
    const password = document.getElementById("login-password").value.trim();
    const statusEl = document.getElementById("login-status");
    statusEl.textContent = "";

    if (!username || !password) {
        statusEl.textContent = "Please enter username and password.";
        statusEl.className = "status-message status-error";
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/user/login`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ username, password })
        });

        const data = await res.json();

        if (!res.ok || !data.token) {
            statusEl.textContent = data.error || "Login failed.";
            statusEl.className = "status-message status-error";
            return;
        }

        authToken = data.token;
        currentUser = username;
        statusEl.textContent = "";

        initSocket();
        enterLobby();

    } catch (err) {
        console.error(err);
        statusEl.textContent = "Network error.";
        statusEl.className = "status-message status-error";
    }
}


// =========================
// Lobby
// =========================

function showView(id) {
    document.getElementById("auth-view").classList.add("hidden");
    document.getElementById("lobby-view").classList.add("hidden");
    document.getElementById("game-view").classList.add("hidden");

    document.getElementById(id).classList.remove("hidden");
}

async function enterLobby() {
    // Update user info
    document.getElementById("display-username").textContent = currentUser || "-";

    // Fetch current points
    await refreshPoints();

    // Clear status display
    document.getElementById("lobby-status").textContent = "";
    document.getElementById("pvp-panel").classList.add("hidden");

    showView("lobby-view");
}

async function refreshPoints() {
    const pointsEl = document.getElementById("display-points");
    pointsEl.textContent = "-";

    if (!authToken) return;

    try {
        const res = await fetch(`${API_BASE}/user/points`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${authToken}`
            }
        });

        const data = await res.json();

        if (res.ok && data.points !== undefined && data.points !== null) {
            pointsEl.textContent = data.points;
        } else {
            pointsEl.textContent = "-";
        }

    } catch (err) {
        console.error(err);
        pointsEl.textContent = "-";
    }
}

function setupLobbyButtons() {
    const btnLogout = document.getElementById("btn-logout");
    const btnVsBot = document.getElementById("btn-vs-bot");
    const btnVsPlayer = document.getElementById("btn-vs-player");
    const btnCreateTable = document.getElementById("btn-create-table");
    const btnJoinTableId = document.getElementById("btn-join-table-id");

    btnLogout.addEventListener("click", () => {
        authToken = null;
        currentUser = null;
        selectedCards.clear();
        
        showView("auth-view");
    });

    btnVsBot.addEventListener("click", () => {
        const lobbyStatus = document.getElementById("lobby-status");

        createTablePve();
    });

    btnVsPlayer.addEventListener("click", () => {
        const pvpPanel = document.getElementById("pvp-panel");
        pvpPanel.classList.remove("hidden");
    });

    btnCreateTable.addEventListener("click", () => {
        createTablePvp();
    });

    btnJoinTableId.addEventListener("click", () => {
        const input = document.getElementById("join-gameid-input");
        const id = input.value.trim();
        if (!id) return;
        joinTableById(id);
    });
}

function initSocket() {

    socket = io("http://50.18.67.156:5001", {
        query: { token: authToken }
    });

    socket.on("system", (msg) => {
        console.log("Broadcast received:", msg);

        if (typeof msg.join !== "undefined") {
            console.log("Player joined from socket:", msg.join);
            return; 
        }

        if (
            (msg.msg === "Player joined game" || msg.msg === "Player 2 joined room") &&
            msg.p1 !== "NA" &&
            msg.p2 !== "NA" &&
            gameMode !== "pvp" 
        ) {
            console.log("Both players ready → startPvpGame");
            round = Number(msg.round) || 1;

            startPvpGame();

            return;
        }


        if (Number(msg.round) !== Number(round)) {
            console.log(
                "Ignore system msg: round mismatch",
                "msg.round =", msg.round,
                "round =", round
            );
            return;
        }
        if (msg.p1 === "NA" && msg.p2 === "NA") {
            console.log("Ignore system msg: both p1/p2 are NA");
            return;
        }
        const hasHandData = msg.p1_hand !== "NA" || msg.p2_hand !== "NA";
        if (!hasHandData) {
            console.log(
                "Ignore system msg: no hand data",
                "p1_hand =", msg.p1_hand,
                "p2_hand =", msg.p2_hand
            );
            return;
        }

        //console.log("Processing round result broadcast...", msg);

        const resultEl = document.getElementById("game-result");

        const iAmP1 = (player === 1);

        const myConfirm  = iAmP1 ? msg.p1_confirm : msg.p2_confirm;
        const oppConfirm = iAmP1 ? msg.p2_confirm : msg.p1_confirm;

        /*
        console.log("Confirm debug:", {
            player,
            iAmP1,
            rawP1: msg.p1_confirm,
            rawP2: msg.p2_confirm,
            myConfirm,
            oppConfirm,
        });
        */

        if (!myConfirm || !oppConfirm) {
            return;
        }

        if (iAmP1) {
            playerHand   = msg.p1_hand;
            opponentHand = msg.p2_hand;
            playerScore   = msg.p1_score;
            opponentScore = msg.p2_score;
        } else {
            playerHand   = msg.p2_hand;
            opponentHand = msg.p1_hand;
            playerScore   = msg.p2_score;
            opponentScore = msg.p1_score;
        }

        renderHand("player-cards", playerHand, true);
        renderHand("opponent-cards", opponentHand, true);

        document.getElementById("score-player").textContent = playerScore;
        document.getElementById("score-opponent").textContent = opponentScore;

        // BO3
        if (Number(playerScore) === 2 || Number(opponentScore) === 2) {
            matchFinished = true;
            document.getElementById("btn-next-round").classList.add("hidden");

            if (playerScore > opponentScore) {
                resultEl.textContent = "You win the match!";
            } else {
                resultEl.textContent = "You lose the match.";
            }

            return;

        } else {    
            document.getElementById("btn-next-round").classList.remove("hidden");
            resultEl.textContent = `Click "Next round" to continue.`;
        }

    });
}


async function createTablePvp() {
    const lobbyStatus = document.getElementById("lobby-status");

    try {
        const res = await fetch(`${API_BASE}/game/creategame`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${authToken}`,
                "mode": "True"     // True = pvp, False = pve
            }
        });

        const data = await res.json();

        if (!res.ok) {
            lobbyStatus.textContent = data.error || "Create game failed.";
            lobbyStatus.className = "status-message status-error";
            return;
        }

        gameId = data.gameid;
        isTableHost = true;
        player = 1; 
        round = 1;
 
        lobbyStatus.textContent =
            `Table #${gameId} created. Waiting for another player to join. ` +
            `Share this table id with your friend.`;
        lobbyStatus.className = "status-message status-ok";

        //console.log("emit listen", { gameId, player });
        socket.emit("listen", {
            //token: localStorage.getItem("jtoken"),
            gameid: gameId,
            player_id: player
        });

    } catch (err) {
        console.error(err);
        lobbyStatus.textContent = "Network error when creating game.";
        lobbyStatus.className = "status-message status-error";
    }
}

async function joinTableById(id) {
    const lobbyStatus = document.getElementById("lobby-status");

    try {
        const res = await fetch(`${API_BASE}/game/joingame`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${authToken}`,
                "gameid": id
            }
        });

        const data = await res.json();

        if (!res.ok) {
            lobbyStatus.textContent = data.error || "Join game failed.";
            lobbyStatus.className = "status-message status-error";
            return;
        }

        gameId = id;
        isTableHost = false;
        player = 2; 
        round = 1;

        lobbyStatus.textContent = `Joined table #${gameId}. Preparing first round...`;
        lobbyStatus.className = "status-message status-ok";

        //console.log("emit listen", { gameId, player });
        socket.emit("listen", {
            //token: localStorage.getItem("jtoken"),
            gameid: gameId,
            player_id: player
        });

    } catch (err) {
        console.error(err);
        lobbyStatus.textContent = "Network error when joining game.";
        lobbyStatus.className = "status-message status-error";
    }
}

async function startPvpGame() {
    gameMode = "pvp";

    playerScore = 0;
    opponentScore = 0;

    matchFinished = false;
    gameFinished = false;

    if (!gameId || !round) {
        console.error("PVP state missing.");
        return;
    }

    showView("game-view");

    const resultEl = document.getElementById("game-result");

    try {
        const res = await fetch(`${API_BASE}/game/track`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                rid: `${gameId}#${round}`,
                player: player
            })
        });

        const data = await res.json();

        if (!res.ok) {
            // Host (player 1) usually enters here when player 2 has not joined yet (Round not found 404).
            if (player === 1) {
                // If this is player 1, retry every 2 seconds.
                resultEl.textContent =
                    data.error || "Waiting for opponent to join...";

                playerHand = [];
                opponentHand = [0, 0, 0, 0, 0]; 

                renderGameView();
                
                setTimeout(startPvpGame, 2000);

            } else {
                // If this is player 2, just show the error directly.
                resultEl.textContent = data.error || "Start game failed.";
            }

            return;
        }

        playerHand = data.handdeck; 
        redrawRemaining = data.redraw;
        selectedCards.clear();
        
        opponentHand = [0, 0, 0, 0, 0];  

        document.getElementById("redraw-count").textContent = redrawRemaining;
        document.getElementById("opponent-cards").innerHTML = "";
        document.getElementById("btn-next-round").classList.add("hidden");

        renderGameView();

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error when tracking game.";
    }
}


async function createTablePve() {
    const lobbyStatus = document.getElementById("lobby-status");

    try {
        const res = await fetch(`${API_BASE}/game/creategame`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${authToken}`,
                "mode": "False"     // True = pvp, False = pve
            }
        });

        const data = await res.json();

        if (!res.ok) {
            lobbyStatus.textContent = data.error || "Create game failed.";
            lobbyStatus.className = "status-message status-error";
            return;
        }

        gameId = data.gameid;
        isTableHost = true;
        player = 1; 
        round = 1;
 
        lobbyStatus.textContent = `Table #${gameId} created. Starting match vs bot... `;
        lobbyStatus.className = "status-message status-ok";

        await startPveGame();

    } catch (err) {
        console.error(err);
        lobbyStatus.textContent = "Network error when creating game.";
        lobbyStatus.className = "status-message status-error";
    }
}

async function startPveGame() {

    gameMode = "pve";

    round = 1
    playerScore = 0;
    opponentScore = 0;

    matchFinished = false;
    gameFinished = false;

    if (gameId == null) { 
        console.error("PVE state missing: gameId not set.");
        return;
    }
    if (typeof round !== "number" || Number.isNaN(round)) {
        console.error("PVE state missing: invalid round:", round);
        return;
    }

    showView("game-view");

    const resultEl = document.getElementById("game-result");

    try {
        const res = await fetch(`${API_BASE}/game/track`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                rid: `${gameId}#${round}`,
                player: player
            })
        });

        const data = await res.json();

        if (!res.ok) {
            if (!res.ok) {
                resultEl.textContent = data.error || "Failed to load starting hand.";
                return;
            }
        }

        playerHand = data.handdeck; 
        redrawRemaining = data.redraw; 
        selectedCards.clear();
        
        opponentHand = [0, 0, 0, 0, 0];

        document.getElementById("redraw-count").textContent = redrawRemaining;
        document.getElementById("opponent-cards").innerHTML = "";
        document.getElementById("btn-next-round").classList.add("hidden");

        renderGameView();

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error when tracking game.";
    }
}


// =========================
// Game view
// =========================

function setupGameButtons() {
    const btnRedraw = document.getElementById("btn-redraw");
    const btnConfirm = document.getElementById("btn-confirm");
    const btnNextRound = document.getElementById("btn-next-round");
    const btnBackLobby = document.getElementById("btn-back-lobby");
  
    btnRedraw.addEventListener("click", () => {
        if (gameFinished || matchFinished) return;
        handleRedraw();
    });

    btnConfirm.addEventListener("click", () => {
        if (gameFinished || matchFinished) return;
        if (gameMode === "pvp") {
            return handleConfirmPvp();
        } else {
            return handleConfirmPve();
        }
    });

    btnNextRound.addEventListener("click", () => {
        if (!gameFinished || matchFinished) return;
        if (round >= MAX_ROUNDS) return;

        startNextRound();
    });

    btnBackLobby.addEventListener("click", () => {
        selectedCards.clear();

        playerScore = 0;
        opponentScore = 0;
        playerHand = [];
        opponentHand = [];
        gameFinished = false;
        matchFinished = false;

        gameMode = null;
        gameId = null;
        round = 1;
        rId = null;

        document.getElementById("game-result").textContent = "";
        document.getElementById("opponent-cards").innerHTML = "";
        document.getElementById("btn-next-round").classList.add("hidden");

        enterLobby();
    });
}


async function handleRedraw() {
    const resultEl = document.getElementById("game-result");

    rId = `${gameId}#${round}`;
    const indexes = Array.from(selectedCards).sort((a, b) => a - b);

    if (!rId || !player) {
        resultEl.textContent = "Game state missing.";
        return;
    }
    if (redrawRemaining <= 0) {
        resultEl.textContent = "You have no redraws left.";
        return;
    }
    if (selectedCards.size === 0) {
        resultEl.textContent = "Select at least one card to redraw.";
        return;
    }
    
    try {
        const res = await fetch(`${API_BASE}/game/redraw`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                rid: rId,
                player: player,
                redraw_positions: indexes
            })
        });

        const data = await res.json();

        if (!res.ok) {
            resultEl.textContent = data.error || "Redraw failed.";
            return;
        }

        playerHand = data.new_hand;
        redrawRemaining = data.remaining_redraws;
        selectedCards.clear();

        document.getElementById("redraw-count").textContent = redrawRemaining;
        renderHand("player-cards", playerHand, true);

        resultEl.textContent = "Redraw completed.";
    
    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error during redraw.";
    }
}


async function handleConfirmPvp() {

    renderHand("player-cards", playerHand, true);
    renderHand("opponent-cards", opponentHand, false);

    const resultEl = document.getElementById("game-result");
    resultEl.textContent = "Hand confirmed. Waiting for opponent to confirm...";

    rId = `${gameId}#${round}`;

    if (!rId || !player) {
        resultEl.textContent = "Game state missing.";
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/game/confirm`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                rid: rId,
                player: player
            })
        });

        const data = await res.json();

        if (!res.ok) {
            resultEl.textContent = data.error || "Confirm failed.";
            return;
        }

        gameFinished = true; 

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error during confirm.";
    }
}

async function handleConfirmPve() {

    gameFinished = true; 

    renderHand("player-cards", playerHand, true);
    renderHand("opponent-cards", opponentHand, false);

    const resultEl = document.getElementById("game-result");
    
    rId = `${gameId}#${round}`;

    if (!rId || !player) {
        resultEl.textContent = "Game state missing.";
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/game/confirm`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({ rid: rId, player })
        });

        const data = await res.json();

        if (!res.ok) {
            resultEl.textContent = data.error || "Confirm failed.";
            return;
        }

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error during confirm.";
    }

    try {
        const res = await fetch(`${API_BASE}/game/track`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                rid: `${gameId}#${round}`,
                player: 2
            })
        });

        const data = await res.json();

        if (!res.ok) {
            resultEl.textContent = data.error || "Round not ready yet.";
            return;
        }

        opponentHand = data.handdeck;
        playerScore   = data.p1_score;
        opponentScore = data.p2_score;

        renderHand("opponent-cards", opponentHand, true);

        document.getElementById("score-player").textContent = playerScore;
        document.getElementById("score-opponent").textContent = opponentScore;

        // BO3
        if (Number(playerScore) === 2 || Number(opponentScore) === 2) {
            matchFinished = true;
            document.getElementById("btn-next-round").classList.add("hidden");
            resultEl.textContent = playerScore > opponentScore
                ? "You win the match!"
                : "You lose the match.";

            return;

        } else {
            document.getElementById("btn-next-round").classList.remove("hidden");
            resultEl.textContent = `Click "Next round" to continue.`;
        }

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error when tracking game.";
    }
}


async function startNextRound() {
    if (round >= MAX_ROUNDS) {
        return;
    }

    round += 1;
    gameFinished = false;

    const resultEl = document.getElementById("game-result");
    resultEl.textContent = `Loading round ${round}...`;

    try {
        const res = await fetch(`${API_BASE}/game/track`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                rid: `${gameId}#${round}`,
                player: player
            })
        });

        const data = await res.json();

        if (!res.ok) {
            resultEl.textContent = data.error || "Round not ready yet.";
            return;
        }

        playerHand = data.handdeck;  
        redrawRemaining = data.redraw;
        selectedCards.clear();
        
        opponentHand = [0, 0, 0, 0, 0];  

        document.getElementById("redraw-count").textContent = redrawRemaining;
        document.getElementById("opponent-cards").innerHTML = "";
        
        document.getElementById("btn-next-round").classList.add("hidden");

        document.getElementById("game-result").textContent = "";

        renderGameView();

    } catch (err) {
        console.error(err);
        resultEl.textContent = "Network error when tracking game.";
    }
}


// =========================

function renderGameView() {
    const modeLabel = document.getElementById("game-mode-label");
    const tableLabel = document.getElementById("game-table-label");
    const redrawCountEl = document.getElementById("redraw-count");
    const resultEl = document.getElementById("game-result");

    if (gameMode === "pvp") {
        modeLabel.textContent = "vs Player";
    } else {
        modeLabel.textContent = "vs Bot";
    }

    tableLabel.textContent = gameId || "-";
    redrawCountEl.textContent = redrawRemaining;

    document.getElementById("game-round-label").textContent = round;
    document.getElementById("game-round-max").textContent = MAX_ROUNDS;
    document.getElementById("score-player").textContent = playerScore;
    document.getElementById("score-opponent").textContent = opponentScore;

    renderHand("player-cards", playerHand, true);
    renderHand("opponent-cards", opponentHand, false);
}

function renderHand(containerId, hand, isPlayer) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    hand.forEach((cardValue, index) => {
        const { rank, suit, isRed } = getCardInfo(cardValue);

        const cardDiv = document.createElement("div");
        cardDiv.className = "card";
        if (isRed) cardDiv.classList.add("red");

        if (isPlayer && selectedCards.has(index)) {
            cardDiv.classList.add("selected");
        }

        const top = document.createElement("div");
        top.className = "card-rank";
        top.textContent = rank;

        const center = document.createElement("div");
        center.className = "card-suit";
        center.textContent = suit;

        const bottom = document.createElement("div");
        bottom.className = "card-corner-bottom";
        bottom.textContent = `${rank}${suit}`;

        cardDiv.appendChild(top);
        cardDiv.appendChild(center);
        cardDiv.appendChild(bottom);

        if (isPlayer) {
            cardDiv.addEventListener("click", () => {
                toggleCardSelection(index);
            });
        } else {
            if (!gameFinished) {
                cardDiv.style.background = "linear-gradient(145deg, #3c4c5f, #1b2838)";
                cardDiv.style.color = "#ecf0f1";
                top.textContent = "?";
                center.textContent = "★";
                bottom.textContent = "?";
            }
        }

        container.appendChild(cardDiv);
    });
}

function getCardInfo(value) {
    const suitIndex = Math.floor(value / 13); // 0..3
    const rankIndex = value % 13;            // 0..12

    const suit = SUITS[suitIndex];
    const rank = RANKS[rankIndex];

    const isRed = suit === "♦" || suit === "♥";

    return { rank, suit, isRed };
}

function toggleCardSelection(index) {
    if (selectedCards.has(index)) {
        selectedCards.delete(index);
    } else {
        selectedCards.add(index);
    }
    
    renderHand("player-cards", playerHand, true);
}

