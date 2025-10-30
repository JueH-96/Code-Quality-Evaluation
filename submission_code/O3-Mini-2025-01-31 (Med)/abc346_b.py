def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    # Read inputs
    W = int(data[0])
    B = int(data[1])
    total = W + B  # total length of desired substring

    # The infinite string S is made up by infinitely repeating this pattern:
    pattern = "wbwbwwbwbwbw"  # period length is 12
    period = len(pattern)

    # To guarantee that every substring of length `total` (which might cross a pattern boundary)
    # appears in our simulation, we simulate enough characters.
    # It is enough to simulate substring with length "total + period".
    repeats = (total // period) + 2  # +2 ensures we cover enough extra characters.
    s = pattern * repeats
    n = len(s)

    # Build prefix sum array for counts of 'w'
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + (1 if s[i] == 'w' else 0)

    # Check over all contiguous substrings of length total for correct counts.
    found = False
    for i in range(n - total + 1):
        cnt_w = prefix[i + total] - prefix[i]
        if cnt_w == W:
            found = True
            break

    sys.stdout.write("Yes" if found else "No")

if __name__ == '__main__':
    main()