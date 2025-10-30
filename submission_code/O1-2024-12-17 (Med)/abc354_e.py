def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    A = [0]*N
    B = [0]*N
    idx = 1
    for i in range(N):
        A[i] = int(input_data[idx]); B[i] = int(input_data[idx+1])
        idx += 2

    # If N=1, Takahashi cannot make a move, so Aoki wins immediately.
    if N == 1:
        print("Aoki")
        return
    
    # Precompute whether a pair (i, j) can be removed (i < j).
    can_remove = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if A[i] == A[j] or B[i] == B[j]:
                can_remove[i][j] = True
                can_remove[j][i] = True

    # Prepare a memo array for DP over subsets (bitmask).
    # dp[mask] = None => not computed; True => winning position; False => losing position
    dp = [None] * (1 << N)

    def rec(mask):
        # If we already know the result for this mask, return it.
        if dp[mask] is not None:
            return dp[mask]
        
        # Try all pairs (i, j) with i < j to remove
        # If any move leads to a losing position for the opponent, this is winning.
        m = mask
        # Optimization: if fewer than 2 cards in mask, it's losing for the current player anyway
        if bin(mask).count('1') < 2:
            dp[mask] = False
            return False

        # We'll iterate over pairs that are still in mask
        # If can_remove[i][j] is True and both i,j are in mask, then we can move to next_mask
        # If rec(next_mask) == False => dp[mask] = True
        # If we exhaust all pairs and can't find such a next_mask => dp[mask] = False
        card_indices = []
        # We can gather the set bits quickly
        # but let's do a simpler approach:
        for i in range(N):
            if (m >> i) & 1:
                card_indices.append(i)
        
        sz = len(card_indices)
        for p in range(sz):
            i = card_indices[p]
            for q in range(p+1, sz):
                j = card_indices[q]
                if can_remove[i][j]:
                    next_mask = mask ^ (1 << i) ^ (1 << j)
                    if not rec(next_mask):
                        dp[mask] = True
                        return True
        
        dp[mask] = False
        return False

    # Compute and output the result for the full set of cards.
    full_mask = (1 << N) - 1
    if rec(full_mask):
        print("Takahashi")
    else:
        print("Aoki")

# Do not forget to call main().
if __name__ == "__main__":
    main()