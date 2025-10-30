# YOUR CODE HERE
# Read N and X
N, X = map(int, input().split())

# Read the scores
scores = list(map(int, input().split()))

# Calculate sum of scores that are X or less
total = sum(score for score in scores if score <= X)

# Print the result
print(total)