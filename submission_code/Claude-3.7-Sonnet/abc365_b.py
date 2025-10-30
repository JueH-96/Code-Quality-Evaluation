# Read the input
N = int(input())
A = list(map(int, input().split()))

# Create a list of (value, position) pairs where position is 1-indexed
indexed_values = [(val, idx + 1) for idx, val in enumerate(A)]

# Sort the pairs in descending order based on values
indexed_values.sort(reverse=True)

# Print the 1-indexed position of the second largest element
print(indexed_values[1][1])