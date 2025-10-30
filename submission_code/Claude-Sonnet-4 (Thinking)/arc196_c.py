from math import factorial

def solve():
    N = int(input())
    S = input().strip()
    
    # Find positions of W's and B's (1-indexed)
    W = [i+1 for i in range(2*N) if S[i] == 'W']
    B = [i+1 for i in range(2*N) if S[i] == 'B']
    
    # Total number of pairings
    total = factorial(N)
    
    # Use inclusion-exclusion
    excluded = 0
    for mask in range(1, 1 << (2*N-1)):
        subset = [k+1 for k in range(2*N-1) if mask & (1 << k)]
        
        # Compute union of white vertices > k and black vertices <= k for k in subset
        W_union = set()
        B_union = set()
        
        for k in subset:
            W_k_plus = set(w for w in W if w > k)
            B_k_minus = set(b for b in B if b <= k)
            W_union |= W_k_plus
            B_union |= B_k_minus
        
        # Check if it's possible to avoid all conditions in subset
        if len(W_union) + len(B_union) <= N:
            intersection_size = factorial(N - len(B_union)) * factorial(N - len(W_union)) // factorial(N - len(W_union) - len(B_union))
        else:
            intersection_size = 0
        
        if len(subset) % 2 == 1:
            excluded += intersection_size
        else:
            excluded -= intersection_size
    
    answer = (total - excluded) % 998244353
    print(answer)

solve()