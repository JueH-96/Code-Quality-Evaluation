import sys
import random

def main():
    """
    Solves the problem by comparing multisets of subsequences using randomized hashing.
    The main logic is encapsulated in this function.
    """
    # Set up fast I/O
    input = sys.stdin.readline
    
    # Read problem size and number of queries
    N, Q = map(int, input().split())
    
    # Read the two sequences
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # --- Precomputation using Randomized Hashing ---

    # Using two hash functions to minimize collision probability.
    # H1, H2 store random 64-bit hash values for each integer from 1 to N.
    # We use 1-based indexing for these hash arrays for convenience, matching problem's indexing.
    H1 = [0] * (N + 1)
    H2 = [0] * (N + 1)
    for i in range(1, N + 1):
        H1[i] = random.getrandbits(64)
        H2[i] = random.getrandbits(64)

    # Precompute prefix sums of hashes for sequence A.
    # PA1[i] stores the sum of H1 hashes for the first i elements of A.
    # PA2[i] stores the sum of H2 hashes for the first i elements of A.
    PA1 = [0] * (N + 1)
    PA2 = [0] * (N + 1)
    for i in range(N):
        val = A[i]
        PA1[i + 1] = PA1[i] + H1[val]
        PA2[i + 1] = PA2[i] + H2[val]

    # Precompute prefix sums of hashes for sequence B.
    PB1 = [0] * (N + 1)
    PB2 = [0] * (N + 1)
    for i in range(N):
        val = B[i]
        PB1[i + 1] = PB1[i] + H1[val]
        PB2[i + 1] = PB2[i] + H2[val]

    # --- Process Queries ---

    answers = []
    for _ in range(Q):
        l, r, L, R = map(int, input().split())

        # Two subsequences can be rearranged to match if and only if:
        # 1. They have the same length.
        # 2. They are anagrams (i.e., have the same multiset of elements).

        # 1. Check lengths
        if r - l != R - L:
            answers.append("No")
            continue

        # 2. Check for multiset equality using our precomputed hashes.
        # A range hash is computed in O(1) using prefix sums.
        # hash(A[l..r]) = prefix_sum[r] - prefix_sum[l-1]
        
        # Calculate hashes for the subsequence of A
        hash_A1 = PA1[r] - PA1[l - 1]
        hash_A2 = PA2[r] - PA2[l - 1]

        # Calculate hashes for the subsequence of B
        hash_B1 = PB1[R] - PB1[L - 1]
        hash_B2 = PB2[R] - PB2[L - 1]

        # If both pairs of hashes match, the multisets are the same with very high probability.
        if hash_A1 == hash_B1 and hash_A2 == hash_B2:
            answers.append("Yes")
        else:
            answers.append("No")
    
    # Print all answers, each on a new line.
    print("
".join(answers))

if __name__ == "__main__":
    main()