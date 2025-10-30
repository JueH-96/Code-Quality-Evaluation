# YOUR CODE HERE
import sys

# Read the number of integers, N
# Use sys.stdin.readline for potentially faster input reading
n = int(sys.stdin.readline())

# Read the N integers A_1, A_2, ..., A_N into a list
# Use sys.stdin.readline and map to int
a = list(map(int, sys.stdin.readline().split()))

# The problem asks for the largest integer among those that are not the largest overall.
# This is equivalent to finding the second largest distinct value in the list.

# Step 1: Find the unique elements in the list A.
# Using a set efficiently removes duplicate values.
unique_elements = set(a)

# Step 2: Convert the set of unique elements back into a list.
# This allows us to sort the elements.
unique_list = list(unique_elements)

# Step 3: Sort the list of unique elements in descending order.
# The largest element will be at index 0, the second largest at index 1, and so on.
unique_list.sort(reverse=True)

# Step 4: Access the second element of the sorted list.
# The problem constraints guarantee that N >= 2 and not all elements are equal.
# This ensures that there are at least two distinct elements in the input,
# so the list `unique_list` will have at least two elements, and index 1 is valid.
# This second element is the largest value among those that are not the absolute largest.
result = unique_list[1]

# Step 5: Print the result to standard output.
print(result)