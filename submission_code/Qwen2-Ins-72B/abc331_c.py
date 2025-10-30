from bisect import bisect_left, bisect_right
from itertools import accumulate

N = int(input())
A = list(map(int, input().split()))

# Create a list of unique elements in A, sorted in ascending order
unique_elements = sorted(set(A))

# Create a list that maps each element in A to its index in unique_elements
element_indices = {element: index for index, element in enumerate(unique_elements)}

# Create a list that maps each index in unique_elements to the number of occurrences of that element in A
element_counts = [0] * len(unique_elements)

# Count the occurrences of each element in A
for element in A:
    element_counts[element_indices[element]] += 1

# Create a list that maps each index in unique_elements to the sum of all elements in A that are greater than the element at that index
sums = [0] * len(unique_elements)
for i in range(len(unique_elements) - 2, -1, -1):
    sums[i] = sums[i + 1] + unique_elements[i + 1] * element_counts[i + 1]

# Create a list that maps each index in unique_elements to the sum of all elements in A that are greater than the element at that index, taking into account the number of occurrences of that element
sums_with_counts = list(accumulate(sums))

# For each element in A, print the sum of all elements in A that are greater than that element
for element in A:
    index = element_indices[element]
    print(sums_with_counts[index] - (sums[index] - unique_elements[index] * element_counts[index]), end=' ')