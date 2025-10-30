# YOUR CODE HERE
import sys

def find_triple(N, X, A):
    # Create a list of tuples (value, index) to keep track of original indices
    indexed_A = [(A[i], i+1) for i in range(N)]
    # Sort the list based on the value
    indexed_A.sort()
    
    # Iterate through the list with three pointers
    for i in range(N-2):
        left = i + 1
        right = N - 1
        while left < right:
            current_sum = indexed_A[i][0] + indexed_A[left][0] + indexed_A[right][0]
            if current_sum == X:
                # Get the original indices in sorted order
                indices = sorted([indexed_A[i][1], indexed_A[left][1], indexed_A[right][1]])
                return indices
            elif current_sum < X:
                left += 1
            else:
                right -= 1
    return -1

# Read input
N, X = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Find the triple
result = find_triple(N, X, A)

# Print the result
if result == -1:
    print(-1)
else:
    print(result[0], result[1], result[2])