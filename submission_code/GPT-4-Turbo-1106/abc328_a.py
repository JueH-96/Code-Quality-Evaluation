# Read the input values
N, X = map(int, input().split())
S = list(map(int, input().split()))

# Calculate the total score for problems with a score of X or less
total_score = sum(score for score in S if score <= X)

# Print the answer
print(total_score)