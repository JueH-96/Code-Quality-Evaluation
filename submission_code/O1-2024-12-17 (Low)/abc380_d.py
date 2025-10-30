def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    S = data[0]
    n = len(S)
    Q = int(data[1])
    K = list(map(int, data[2:]))

    # For flipping the case of a single character (we assume input is always A-Z or a-z).
    def flip_char(c):
        # Since we only have letters, toggling the 6th bit of ASCII code flips case.
        return chr(ord(c) ^ 32)

    # We'll convert S into a list (for fast indexing).
    s_list = list(S)

    # Function to get the character at 1-based index k after "infinite" expansions.
    def get_char(k):
        flip_count = 0
        # p will be the smallest power-of-two multiple of n that is >= k
        p = n
        while p < k:
            p <<= 1
        
        # We'll keep halving p until it becomes n
        while p > n:
            half = p >> 1
            if k > half:
                k -= half
                flip_count ^= 1  # Toggle whether we need to flip
            p = half
        
        # Now k <= n, so the character is from the original S
        c = s_list[k-1]
        if flip_count == 1:
            c = flip_char(c)
        return c

    out = []
    for pos in K:
        out.append(get_char(pos))
    print(" ".join(out))

# Don't forget to call main().
if __name__ == "__main__":
    main()