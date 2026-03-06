def decode(card:int):
    suit = ['S', 'H', 'D', 'C']
    num = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    s = suit[card // 13]
    n = num[card % 13]
    return f"{n}{s}"