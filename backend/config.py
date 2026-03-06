import boto3

dynamodb = boto3.resource(
    "dynamodb",
    region_name="us-west-1"
)

JWT_SECRET = "ee547_group_6"

db_user = dynamodb.Table("users")
db_counter = dynamodb.Table("counter")
db_game = dynamodb.Table("games")
db_rounds = dynamodb.Table("rounds")
db_deck = dynamodb.Table("deck")
db_training_queue = dynamodb.Table("training_queue")