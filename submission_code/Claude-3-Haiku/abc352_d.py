from typing import List

def find_min_length(N: int, K: int, P: List[int]) -> int:
    # Create a dictionary to store the indices of each number
    indices = {num: i for i, num in enumerate(P)}
    
    # Initialize the minimum length to N
    min_length = N
    
    # Iterate through all possible good index sequences
    for i in range(N - K + 1):
        # Check if the subsequence is a rearrangement of consecutive integers
        if all(P[i + j] == i + j + 1 for j in range(K)):
            # Update the minimum length if necessary
            min_length = min(min_length, K)
    
    return min_length

# Read the input
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Solve the problem and print the answer
print(find_min_length(N, K, P))