# Read N and L
N, L = map(int, input().split())

# Read the scores
scores = list(map(int, input().split()))

# Count how many scores are >= L
count = 0
for score in scores:
    if score >= L:
        count += 1

# Print the count
print(count)