# Read the input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Function to find the maximum XOR of K distinct elements
def max_xor(A, K):
    # Sort the array in descending order
    A.sort(reverse=True)
    
    # Initialize the maximum XOR to 0
    max_xor = 0
    
    # Iterate through the first K elements
    for i in range(K):
        # Update the maximum XOR
        max_xor ^= A[i]
    
    return max_xor

# Print the answer
print(max_xor(A, K))