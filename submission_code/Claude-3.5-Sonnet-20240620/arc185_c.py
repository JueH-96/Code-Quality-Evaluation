# YOUR CODE HERE
def find_triple(N, X, A):
    # Create a dictionary to store the complement of each pair sum
    complement_dict = {}
    
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            pair_sum = A[i] + A[j]
            complement = X - pair_sum
            
            # Check if the complement exists in the remaining part of the array
            if complement in A[j+1:]:
                k = A.index(complement, j + 1)
                return i + 1, j + 1, k + 1
            
            # Store the complement for future checks
            complement_dict[pair_sum] = (i, j)
    
    return -1

# Read input
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Find the triple
result = find_triple(N, X, A)

# Print the result
if result == -1:
    print(-1)
else:
    print(*result)