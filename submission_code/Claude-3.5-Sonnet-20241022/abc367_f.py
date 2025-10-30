def can_rearrange(A, B, l, r, L, R):
    # Get subsequences
    subA = A[l-1:r]
    subB = B[L-1:R]
    
    # If lengths are different, impossible
    if len(subA) != len(subB):
        return False
        
    # Count frequencies
    freq_A = {}
    freq_B = {}
    
    for x in subA:
        freq_A[x] = freq_A.get(x, 0) + 1
    for x in subB:
        freq_B[x] = freq_B.get(x, 0) + 1
        
    # Check if frequencies match
    return freq_A == freq_B

# Read input
N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Process queries
for _ in range(Q):
    l, r, L, R = map(int, input().split())
    print("Yes" if can_rearrange(A, B, l, r, L, R) else "No")