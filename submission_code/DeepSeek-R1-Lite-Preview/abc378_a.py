from collections import Counter

# Read the input as a list of integers
A = list(map(int, input().split()))

# Count the frequency of each color
counts = Counter(A)

# Calculate the total number of operations
operations = sum(count // 2 for count in counts.values())

# Print the result
print(operations)