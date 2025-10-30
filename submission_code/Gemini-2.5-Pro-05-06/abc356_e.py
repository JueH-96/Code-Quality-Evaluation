import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MAX_A_VAL = 0
    if N == 0:
        print(0)
        return
        
    for x_val in A:
        if x_val > MAX_A_VAL:
            MAX_A_VAL = x_val
            
    counts = [0] * (MAX_A_VAL + 1)
    for x_val in A:
        counts[x_val] += 1

    ans = 0
    
    # Part 1: Contribution from pairs (A_i, A_j) where A_i = A_j = v.
    for v_val in range(1, MAX_A_VAL + 1):
        if counts[v_val] >= 2:
            ans += counts[v_val] * (counts[v_val] - 1) // 2
    
    # Part 2: Contribution from pairs (A_i, A_j) where A_i = x and A_j = y, with x != y.
    # Sum is C[x]*C[y]*floor(y/x) for x < y.
    suf_counts = [0] * (MAX_A_VAL + 2) # suf_counts[i] = sum(counts[j] for j >= i)
                                      # suf_counts[MAX_A_VAL+1] = 0 as sentinel
    
    suf_counts[MAX_A_VAL] = counts[MAX_A_VAL]
    for i in range(MAX_A_VAL - 1, 0, -1): # Iterate from MAX_A_VAL-1 down to 1
        suf_counts[i] = suf_counts[i+1] + counts[i]

    s_b = 0
    # Iterate over each value x that is present in A
    # x ranges from 1 up to MAX_A_VAL. If x=MAX_A_VAL, T_x will be 0.
    for x in range(1, MAX_A_VAL + 1):
        if counts[x] == 0:
            continue
        
        term_for_x = 0 # This will be T_x
        
        # Contribution for k=1: sum SufC[max(1*x, x+1)] = SufC[x+1]
        term_for_x += suf_counts[x+1] 
        # Note: if x=MAX_A_VAL, x+1 = MAX_A_VAL+1. suf_counts[MAX_A_VAL+1] is 0.
        
        # Contribution for k >= 2: sum SufC[max(k*x, x+1)] = SufC[kx]
        # Iterate mult = kx from 2x up to MAX_A_VAL
        for mult in range(2 * x, MAX_A_VAL + 1, x):
             term_for_x += suf_counts[mult]
        
        s_b += counts[x] * term_for_x
        
    ans += s_b
    print(ans)

solve()