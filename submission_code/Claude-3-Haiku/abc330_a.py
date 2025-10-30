# Read the input
N, L = map(int, input().split())
scores = list(map(int, input().split()))

# Count the number of people who passed
passed = 0
for score in scores:
    if score >= L:
        passed += 1

# Print the answer
print(passed)