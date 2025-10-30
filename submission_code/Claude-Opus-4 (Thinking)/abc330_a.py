# YOUR CODE HERE
# Read N and L
N, L = map(int, input().split())

# Read the scores
scores = list(map(int, input().split()))

# Count how many people passed
count = 0
for score in scores:
    if score >= L:
        count += 1

# Print the answer
print(count)