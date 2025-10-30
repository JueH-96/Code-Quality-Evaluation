# YOUR CODE HERE
def find_lexicographically_smallest(N, P, A):
    # Create a list of tuples (A_i, i) and sort it by A_i
    indexed_A = sorted((A[i], i) for i in range(N))
    
    # Create a list to store the lexicographically smallest A
    smallest_A = [0] * N
    
    # Fill the smallest_A list using the sorted indexed_A
    for new_index, (value, original_index) in enumerate(indexed_A):
        smallest_A[original_index] = new_index + 1
    
    # Apply the permutation P to the smallest_A
    result = [0] * N
    for i in range(N):
        result[i] = smallest_A[P[i] - 1]
    
    return result

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
A = list(map(int, data[N+1:2*N+1]))

# Find the lexicographically smallest A
result = find_lexicographically_smallest(N, P, A)

# Print the result
print(" ".join(map(str, result)))