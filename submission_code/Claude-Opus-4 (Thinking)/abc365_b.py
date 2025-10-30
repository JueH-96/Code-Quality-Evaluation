# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Find the second largest value
sorted_values = sorted(A, reverse=True)
second_largest_value = sorted_values[1]

# Find the position of the second largest value in the original array
position = A.index(second_largest_value) + 1  # +1 for 1-indexed

print(position)