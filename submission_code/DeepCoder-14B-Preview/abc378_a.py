from collections import Counter

# Read the input as a list of integers
A = list(map(int, input().split()))

# Count the occurrences of each color
counts = Counter(A)

# Calculate the total number of operations
total_operations = sum(v // 2 for v in counts.values())

# Output the result
print(total_operations)