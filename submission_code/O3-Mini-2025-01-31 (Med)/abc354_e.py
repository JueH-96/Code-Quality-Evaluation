def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [0]*N
    B = [0]*N
    for i in range(N):
        A[i] = int(next(it))
        B[i] = int(next(it))
    
    # Precompute all pairs (i, j) which can be removed together according to the rules.
    valid_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i] == A[j] or B[i] == B[j]:
                valid_pairs.append((i, j))
    
    # Use memoization on bitmask states.
    # In a state, the bitmask represents which cards are still on the table.
    # dp(mask) returns True if the current player to move (with state "mask") can force a win.
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(mask):
        # Try every valid pair that is present in mask
        found_move = False
        for i, j in valid_pairs:
            if (mask >> i) & 1 and (mask >> j) & 1:
                found_move = True
                new_mask = mask & ~(1 << i) & ~(1 << j)
                # If this move leaves the opponent in a losing position, then current wins.
                if not dp(new_mask):
                    return True
        # If no valid move exists, current player loses.
        if not found_move:
            return False
        return False

    full_mask = (1 << N) - 1
    if dp(full_mask):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    main()