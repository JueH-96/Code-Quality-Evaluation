# YOUR CODE HERE
def reverse_subarray(N, L, R):
    # Create the initial sequence A
    A = list(range(1, N + 1))
    
    # Reverse the subarray from L to R (1-based index)
    # Convert L and R to 0-based index for Python list
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the resulting sequence
    print(' '.join(map(str, A)))

# Read input
import sys
input = sys.stdin.read
N, L, R = map(int, input().strip().split())

# Call the function with the input values
reverse_subarray(N, L, R)