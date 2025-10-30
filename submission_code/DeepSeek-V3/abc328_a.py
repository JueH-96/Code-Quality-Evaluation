# Read input values
N, X = map(int, input().split())
S = list(map(int, input().split()))

# Calculate the total score for problems with score <= X
total = sum(s for s in S if s <= X)

# Print the result
print(total)