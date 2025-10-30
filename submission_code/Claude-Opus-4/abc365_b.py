# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Create a list of (value, original_position) pairs
# Position is 1-indexed
indexed_a = [(a[i], i + 1) for i in range(n)]

# Sort by value in descending order
indexed_a.sort(reverse=True)

# The second element in the sorted list is the second largest
# Print its original position
print(indexed_a[1][1])