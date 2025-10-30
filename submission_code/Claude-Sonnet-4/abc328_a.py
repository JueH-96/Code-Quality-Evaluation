# YOUR CODE HERE
# Read N and X
N, X = map(int, input().split())

# Read the scores
scores = list(map(int, input().split()))

# Calculate total score for problems with score <= X
total = 0
for score in scores:
    if score <= X:
        total += score

# Print the answer
print(total)