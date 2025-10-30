# YOUR CODE HERE
import sys

def find_second_largest_index(N, A):
    # Find the index of the largest element
    max_index = A.index(max(A))
    
    # Remove the largest element from the list
    A.pop(max_index)
    
    # Find the index of the new largest element (second largest in the original list)
    second_max_index = A.index(max(A))
    
    # Adjust the index to account for the removed element
    if second_max_index >= max_index:
        second_max_index += 1
    
    return second_max_index + 1  # +1 to convert from 0-indexed to 1-indexed

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print the result
print(find_second_largest_index(N, A))