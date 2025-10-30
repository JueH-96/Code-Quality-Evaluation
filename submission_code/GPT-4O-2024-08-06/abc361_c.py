# YOUR CODE HERE
def min_max_difference(N, K, A):
    # Length of the subsequence we need to consider
    length_of_B = N - K
    
    # Initialize the minimum difference to a large number
    min_difference = float('inf')
    
    # Iterate over all possible starting points for a subsequence of length N-K
    for start in range(N - length_of_B + 1):
        # Get the subsequence
        subsequence = A[start:start + length_of_B]
        # Calculate the difference between max and min of this subsequence
        current_difference = max(subsequence) - min(subsequence)
        # Update the minimum difference found
        min_difference = min(min_difference, current_difference)
    
    return min_difference

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

print(min_max_difference(N, K, A))