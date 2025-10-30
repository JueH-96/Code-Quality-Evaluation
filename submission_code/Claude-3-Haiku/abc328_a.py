# Read the input
N, X = map(int, input().split())
scores = list(map(int, input().split()))

# Calculate the total score for problems with score <= X
total_score = sum(score for score in scores if score <= X)

# Print the answer
print(total_score)