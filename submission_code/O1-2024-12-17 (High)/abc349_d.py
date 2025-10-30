def main():
    import sys
    data = sys.stdin.read().strip().split()
    L, R = map(int, data)

    def largest_power_of_two_leq(n):
        """Return the largest power of two <= n (n >= 1)."""
        p = 1
        while p <= n:
            p <<= 1
        return p >> 1

    x = L
    intervals = []
    while x < R:
        if x == 0:
            # 0 is a multiple of all powers of two, so pick the largest that fits
            p = largest_power_of_two_leq(R - x)
        else:
            # p is the largest power of two dividing x
            p = x & -x
            # If p exceeds what's left to reach R, halve it till it fits
            while p > R - x:
                p >>= 1
        intervals.append((x, x + p))
        x += p

    print(len(intervals))
    for l_i, r_i in intervals:
        print(l_i, r_i)

# Call main() at the end
if __name__ == "__main__":
    main()