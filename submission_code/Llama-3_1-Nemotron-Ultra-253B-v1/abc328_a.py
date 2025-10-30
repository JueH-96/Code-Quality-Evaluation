# Read input values
n, x = map(int, input().split())
scores = list(map(int, input().split()))

# Calculate the sum of scores that are less than or equal to X
total = sum(s for s in scores if s <= x)

# Output the result
print(total)