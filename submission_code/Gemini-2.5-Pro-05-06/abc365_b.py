# YOUR CODE HERE
# Read N, the number of elements
N = int(input())
# Read the sequence A as a list of integers
A = list(map(int, input().split()))

# Create a sorted version of A in descending order.
# sorted() returns a new list and does not modify the original list A.
sorted_A_values = sorted(A, reverse=True)

# The second largest value in the sequence is at index 1 (0-indexed)
# of this sorted list. This is guaranteed to exist because N >= 2.
second_largest_value = sorted_A_values[1]

# Find the 0-based index of this second_largest_value in the original list A.
# The list.index(value) method returns the 0-based index of the first
# occurrence of 'value'. Since all elements in A are distinct (as per
# problem constraints), this will uniquely identify the element's position.
original_0_based_index = A.index(second_largest_value)

# The problem asks for the "X-th element", which implies a 1-based index.
# So, we convert the 0-based index to a 1-based index by adding 1.
result_X = original_0_based_index + 1

# Print the 1-based index.
print(result_X)