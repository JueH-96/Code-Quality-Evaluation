def main():
    import sys
    data = sys.stdin.read().strip().split()
    S = data[0]
    n = len(S)
    Q = int(data[1])
    queries = data[2:]
    
    # For each position K, we observe that the final (infinitely expanded) string
    # can be viewed as blocks of length n (the length of S):
    #
    #   Block 0: S (no invert)
    #   Block 1: invert(S)
    #   Block 2: invert(S)
    #   Block 3: S
    #   Block 4: invert(S)
    #   Block 5: S
    #   ...
    #
    # In fact, whether a given block j is inverted or not depends on the parity
    # of the number of set bits of j (the "popcount").  If popcount(j) is even,
    # that block is just S; if odd, it is invert(S).
    #
    # To find the K-th character (1-based index):
    #   1) Identify which block it belongs to: block_index = (K-1) // n
    #   2) Within that block, the offset = (K-1) % n
    #   3) character = S[offset]
    #   4) If popcount(block_index) is odd, invert the character’s case
    #
    # This works because after so many (10^100) expansions, the string is huge,
    # and K <= 10^18 is well within that length.  We only need the parity check of
    # block_index for flipping case.

    result = []
    for i in range(Q):
        K = int(queries[i])
        block_index = (K - 1) // n
        offset = (K - 1) % n
        c = S[offset]
        # bit_count() returns the number of set bits in block_index (Python 3.10+)
        if block_index.bit_count() & 1:
            c = c.swapcase()
        result.append(c)
    
    print(' '.join(result))

# Don't forget to call main at the end
if __name__ == "__main__":
    main()