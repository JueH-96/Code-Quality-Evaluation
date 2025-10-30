from collections import Counter

# Read the input as a list of integers
a = list(map(int, input().split()))

# Count the frequency of each number
counts = Counter(a)

# Calculate the total number of pairs
total = sum(c // 2 for c in counts.values())

# Print the result
print(total)