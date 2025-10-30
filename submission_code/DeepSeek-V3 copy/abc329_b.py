# YOUR CODE HERE
# Read input
N = int(input())
A = list(map(int, input().split()))

# Find the maximum value
max_val = max(A)

# Filter out the maximum value and find the new maximum
filtered_A = [x for x in A if x != max_val]
second_max = max(filtered_A)

# Print the result
print(second_max)