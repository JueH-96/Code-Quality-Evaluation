from typing import List

def smallestBoxForToys(N: int, A: List[int], B: List[int]) -> int:
    A.sort()
    B.sort()
    
    # Check if a toy can fit into any existing box
    for i in range(N):
        if A[i] > B[i]:
            # If a toy can't fit, the smallest box needed is the size of that toy
            return A[i]

    # If all toys can fit, return -1 since no new box needs to be purchased
    return -1

# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve and print the answer
print(smallestBoxForToys(N, A, B))