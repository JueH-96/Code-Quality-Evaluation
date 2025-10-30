# YOUR CODE HERE
# Read N and L
N, L = map(int, input().split())

# Read the scores
scores = list(map(int, input().split()))

# Count how many people passed (scored at least L points)
passed_count = 0
for score in scores:
    if score >= L:
        passed_count += 1

# Output the result
print(passed_count)