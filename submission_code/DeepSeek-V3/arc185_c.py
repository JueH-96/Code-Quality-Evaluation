# YOUR CODE HERE
import sys

def find_triple(N, X, A):
    # Since N can be up to 1e6, we need an efficient solution
    # We can use a hash set to store the elements and check for the third element
    # However, since the order matters (i < j < k), we need to iterate in a specific way
    
    # First, create a list of tuples containing the value and its original index
    indexed_A = [(A[i], i+1) for i in range(N)]
    # Sort the list based on the value
    indexed_A.sort()
    
    # Iterate through all possible pairs (i, j) where i < j
    for i in range(N-2):
        for j in range(i+1, N-1):
            # Calculate the required third element
            required = X - indexed_A[i][0] - indexed_A[j][0]
            # Use binary search to find if the required element exists in the remaining list
            left = j+1
            right = N-1
            while left <= right:
                mid = (left + right) // 2
                if indexed_A[mid][0] == required:
                    # Ensure that the indices are in order
                    if indexed_A[i][1] < indexed_A[j][1] < indexed_A[mid][1]:
                        return (indexed_A[i][1], indexed_A[j][1], indexed_A[mid][1])
                    else:
                        # If not, continue searching
                        break
                elif indexed_A[mid][0] < required:
                    left = mid + 1
                else:
                    right = mid - 1
    return -1

# Read input
N, X = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Find the triple
result = find_triple(N, X, A)

# Output the result
if result == -1:
    print(-1)
else:
    print(result[0], result[1], result[2])