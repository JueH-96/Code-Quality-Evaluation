# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Pyramid Sequence problem.
    Reads input N and sequence A.
    Calculates the maximum possible size k of a Pyramid Sequence
    that can be obtained from A using the allowed operations.
    Prints the maximum size k.
    """
    N = int(sys.stdin.readline())
    # The problem constraints state N >= 1, so we don't need to handle N=0.
    A = list(map(int, sys.stdin.readline().split()))

    # L[i] will store the maximum height k such that the subarray A[i-k+1 .. i] 
    # can form the sequence (1, 2, ..., k) after decreasing values.
    # This represents the potential size of the left half of a pyramid ending at index i.
    L = [0] * N
    
    # Base case: The element A[0] itself can form a sequence (1) if A[0] >= 1.
    # Since the constraints guarantee A[i] >= 1, A[0] can always form (1).
    # The maximum k for the single element subarray A[0] is 1.
    L[0] = 1 
    
    # Compute L[i] for i from 1 to N-1 using dynamic programming.
    for i in range(1, N):
        # To form a sequence (1, ..., k) ending at index i, two conditions must hold:
        # 1. The value A[i] must be at least k (A[i] >= k).
        # 2. The subarray ending at index i-1 must be able to form the sequence (1, ..., k-1).
        # The maximum length k' for the sequence ending at i-1 is given by L[i-1].
        # So, we need k-1 <= L[i-1], which implies k <= L[i-1] + 1.
        # Combining both conditions, the maximum possible k is min(A[i], L[i-1] + 1).
        L[i] = min(A[i], L[i-1] + 1)

    # R[i] will store the maximum height k such that the subarray A[i .. i+k-1]
    # can form the sequence (k, k-1, ..., 1) after decreasing values.
    # This represents the potential size of the right half of a pyramid starting at index i.
    R = [0] * N

    # Base case: The element A[N-1] itself can form a sequence (1) if A[N-1] >= 1.
    # Since A[i] >= 1 is guaranteed, A[N-1] can always form (1).
    # The maximum k for the single element subarray A[N-1] is 1.
    R[N-1] = 1 
    
    # Compute R[i] for i from N-2 down to 0 using dynamic programming.
    for i in range(N-2, -1, -1):
        # To form a sequence (k, ..., 1) starting at index i, two conditions must hold:
        # 1. The value A[i] must be at least k (A[i] >= k).
        # 2. The subarray starting at index i+1 must be able to form the sequence (k-1, ..., 1).
        # The maximum length k' for the sequence starting at i+1 is given by R[i+1].
        # So, we need k-1 <= R[i+1], which implies k <= R[i+1] + 1.
        # Combining both conditions, the maximum possible k is min(A[i], R[i+1] + 1).
        R[i] = min(A[i], R[i+1] + 1)

    # A Pyramid Sequence of size k centered at index p requires:
    # - The left half including the peak (1, ..., k) ending at p. The max size possible is L[p]. So k <= L[p].
    # - The right half including the peak (k, ..., 1) starting at p. The max size possible is R[p]. So k <= R[p].
    # Therefore, the maximum size k of a pyramid centered at p is min(L[p], R[p]).
    # The overall maximum size is the maximum of these values over all possible centers p.
    max_k = 0
    for p in range(N):
        # K_p is the maximum pyramid size that can be centered at index p.
        K_p = min(L[p], R[p])
        # Update the overall maximum size found so far.
        max_k = max(max_k, K_p)

    # Print the final result.
    print(max_k)

# Execute the solve function to process input and print output
solve()