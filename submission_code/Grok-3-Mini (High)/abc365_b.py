# Read input
N = int(input())
A = list(map(int, input().split()))

# Create a list of tuples with value and 1-based index
vals = [(A[i], i + 1) for i in range(N)]

# Sort the list in descending order based on the value
sorted_vals = sorted(vals, key=lambda x: x[0], reverse=True)

# The second element in the sorted list is the second largest, get its index
second_largest_index = sorted_vals[1][1]

# Output the index
print(second_largest_index)