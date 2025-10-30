from collections import Counter

# Read input
N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Process queries
for _ in range(Q):
    l, r, L, R = map(int, input().split())
    subseq_A = A[l-1:r]
    subseq_B = B[L-1:R]

    # Check if the rearranged subsequence of A can match subsequence of B
    if Counter(subseq_A) == Counter(subseq_B):
        print("Yes")
    else:
        print("No")