def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Read input: S is the initial string
    S = data[0]
    n = len(S)
    Q = int(data[1])
    queries = list(map(int, data[2:]))

    # Explanation:
    # After each operation, S evolves as S = S + flip(S) where flip(c) toggles the case.
    # Starting from S₀ = S, after one operation we have:
    #   S₁ = S₀ + flip(S₀)
    # After two operations:
    #   S₂ = S₁ + flip(S₁) = S₀ + flip(S₀) + flip(S₀) + S₀
    # Notice that S₂ consists of 4 blocks of size n: S₀, flip(S₀), flip(S₀), S₀.
    #
    # In general, after many operations (indeed, 10^100 times so that the string becomes infinite),
    # every contiguous block of n characters is either S (original) or flip(S),
    # and the pattern of flips is determined by the binary representation of the block's index.
    #
    # Let k (0-indexed) be the position in the infinite string.
    # Write k = block * n + pos, where 0 <= pos < n.
    # It turns out (by induction) that if the number of 1's in the binary representation
    # of block (i.e. popcount(block)) is odd, then the character at position k is the flip of S[pos],
    # otherwise it is exactly S[pos].
    #
    # Thus, for a query asking the K-th character (with K given 1-indexed), we do:
    #   offset = K - 1
    #   block = offset // n, pos = offset % n
    #   answer = flip(S[pos]) if (popcount(block) % 2 == 1) else S[pos]
    #
    # Python 3.10+ supports int.bit_count() which gives the popcount.

    out = []
    for ki in queries:
        k = ki - 1  # change to 0-indexed
        block = k // n
        pos = k % n
        ch = S[pos]
        if block.bit_count() & 1:  # if odd number of ones, then flip the case
            if ch.islower():
                ch = ch.upper()
            else:
                ch = ch.lower()
        out.append(ch)

    sys.stdout.write(" ".join(out))

if __name__ == '__main__':
    main()