# YOUR CODE HERE
from collections import Counter

N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

for _ in range(Q):
    l, r, L, R = map(int, input().split())
    
    # Extract subsequences (convert to 0-based indexing)
    subseq_A = A[l-1:r]
    subseq_B = B[L-1:R]
    
    # Count frequencies
    freq_A = Counter(subseq_A)
    freq_B = Counter(subseq_B)
    
    # Check if frequencies match
    if freq_A == freq_B:
        print("Yes")
    else:
        print("No")