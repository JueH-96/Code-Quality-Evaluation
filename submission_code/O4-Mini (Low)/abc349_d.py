def main():
    import sys
    data = sys.stdin.read().strip().split()
    L = int(data[0])
    R = int(data[1])

    intervals = []
    x = L
    # While we haven't covered up to R
    while x < R:
        rem = R - x
        # max power of two not exceeding rem
        max_pow_rem = rem.bit_length() - 1

        # trailing zeros of x gives alignment
        if x == 0:
            tz = 60  # x=0 is divisible by all 2^i up to problem limit
        else:
            # x & -x isolates lowest set bit; its bit_length-1 is the exponent
            tz = (x & -x).bit_length() - 1

        # pick the largest block we can align and fit
        i = min(tz, max_pow_rem)
        size = 1 << i

        intervals.append((x, x + size))
        x += size

    # Output
    out = []
    out.append(str(len(intervals)))
    for l, r in intervals:
        out.append(f"{l} {r}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()