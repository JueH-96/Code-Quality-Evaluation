# Read the input values
n, x = map(int, input().split())
scores = list(map(int, input().split()))

# Calculate the total sum of scores that are X or less
total = sum(s for s in scores if s <= x)

# Print the result
print(total)