# YOUR CODE HERE
def reverse_subsequence(N, L, R):
    # Create the initial sequence
    A = list(range(1, N + 1))
    
    # Reverse the subsequence from L to R (1-based index)
    A[L-1:R] = A[L-1:R][::-1]
    
    # Print the modified sequence
    print(" ".join(map(str, A)))

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
L = int(data[1])
R = int(data[2])

# Call the function with the input values
reverse_subsequence(N, L, R)