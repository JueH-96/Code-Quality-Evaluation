# Read N and X
N, X = map(int, input().split())

# Read the scores
scores = list(map(int, input().split()))

# Calculate the sum of all scores that are <= X
total = sum(score for score in scores if score <= X)

# Print the answer
print(total)