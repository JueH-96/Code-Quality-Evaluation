# Read N and X from the first line of input
N, X = map(int, input().split())

# Read the N scores from the second line of input
scores = list(map(int, input().split()))

# Initialize a variable to store the total score of problems with score X or less
total_score = 0

# Iterate through each score in the list of scores
for score in scores:
    # Check if the current score is less than or equal to X
    if score <= X:
        # If it is, add it to the total_score
        total_score += score

# Print the final calculated total score
print(total_score)