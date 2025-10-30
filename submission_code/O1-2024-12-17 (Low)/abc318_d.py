def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # We'll read the input in a triangular form and build a full NxN matrix D
    D = [[0]*N for _ in range(N)]
    
    idx = 1
    for i in range(N-1):
        for j in range(i+1, N):
            w = int(input_data[idx])
            idx += 1
            D[i][j] = w
            D[j][i] = w

    from functools import lru_cache
    
    @lru_cache(None)
    def solve(mask):
        # If mask has 0 or 1 bits set, no edges can be formed
        if mask == 0:
            return 0
        # For speed, we can do a quick check:
        if (mask & (mask - 1)) == 0:
            # mask is a power of two => exactly one bit => can't form any edge
            return 0
        
        # pick the lowest set bit i as a pivot
        i = (mask & -mask)  # lowest set bit of mask
        i_index = i.bit_length() - 1  # convert bit to index
        best = 0
        # Option 1: skip i_index
        best = max(best, solve(mask ^ i))
        
        # Option 2: pair i_index with some other j_index
        remaining = mask ^ i
        # we iterate over set bits in remaining
        sub = remaining
        while sub:
            j = (sub & -sub)
            j_index = j.bit_length() - 1
            best = max(best, solve(remaining ^ j) + D[i_index][j_index])
            sub = sub & (sub - 1)
        
        return best
    
    full_mask = (1 << N) - 1
    answer = solve(full_mask)
    print(answer)

# Don't forget to call main
if __name__ == "__main__":
    main()