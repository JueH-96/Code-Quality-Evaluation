def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    # When rearranging digits we must use them all.
    # Such a square number (possibly with leading zeros) must have exactly N digits.
    # So, if we pad the square with zeroes on the left to have N digits,
    # its sorted digit list must equal the sorted digits of S.
    target = sorted(S)
    
    max_val = 10 ** N
    # We need to consider squares < 10^N.
    # x such that x^2 < 10^N. Using math.isqrt, the maximum x is:
    x_max = math.isqrt(max_val - 1) if max_val > 1 else 0
    
    count = 0
    # Prepare the format string for zero-padding to length N.
    pad_format = "0{}d".format(N)
    
    for x in range(x_max + 1):
        sq = x * x
        s_sq = format(sq, pad_format)
        if sorted(s_sq) == target:
            count += 1

    sys.stdout.write(str(count))


if __name__ == '__main__':
    main()