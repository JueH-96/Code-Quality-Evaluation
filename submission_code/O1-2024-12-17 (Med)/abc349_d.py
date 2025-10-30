def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    L, R = map(int, data)
    
    # We want to partition [L, R) into the fewest segments of the form [x, x+2^k]
    # where x is a multiple of 2^k (i.e., x mod 2^k = 0) and x+2^k <= R.
    # A well-known greedy approach (which is guaranteed minimal) is:
    #
    # 1) If L == 0, pick the largest 2^k that is <= (R - L).
    #    Otherwise, pick p = L & (-L) (the largest power of two dividing L).
    # 2) If p > (R - L), keep halving p until it is <= (R - L).
    # 3) Produce an interval [L, L+p], set L := L + p, and repeat until L == R.
    #
    # This procedure yields the unique minimal partition into "good" segments,
    # each of length 2^k and starting at a multiple of 2^k.
    
    intervals = []
    
    while L < R:
        if L == 0:
            # Special case for L=0, since (0 & -0) = 0 won't help.
            length = R - L
            p = 1 << (length.bit_length() - 1)  # largest power of two <= length
        else:
            # p is the largest power of two dividing L
            p = L & -L
            # Make sure p does not exceed (R - L)
            while p > (R - L):
                p >>= 1
        
        intervals.append((L, L + p))
        L += p
    
    # Output the result
    print(len(intervals))
    for seg in intervals:
        print(seg[0], seg[1])

# Do not forget to call main()
if __name__ == "__main__":
    main()