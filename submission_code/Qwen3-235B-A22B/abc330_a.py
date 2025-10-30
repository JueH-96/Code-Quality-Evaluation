# Read the input values
N, L = map(int, input().split())
scores = list(map(int, input().split()))

# Count how many scores are at least L
count = 0
for score in scores:
    if score >= L:
        count += 1

# Output the result
print(count)