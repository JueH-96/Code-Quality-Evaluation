# Read input values
N, L = map(int, input().split())
A = list(map(int, input().split()))

# Count the number of people who scored at least L points
count = 0
for score in A:
    if score >= L:
        count += 1

# Print the result
print(count)