def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    lost = [False] * (N + 1)
    for _ in range(K):
        a = int(next(it))
        lost[a] = True

    # Build the sorted list of socks.
    # For each color from 1 to N, if a sock is lost then one sock remains,
    # otherwise two socks (one pair) remain.
    socks = []
    for color in range(1, N + 1):
        if lost[color]:
            socks.append(color)
        else:
            socks.append(color)
            socks.append(color)
    socks.sort()
    s = len(socks)

    # When s is even, just pair adjacent socks.
    if s % 2 == 0:
        total = 0
        for i in range(0, s, 2):
            total += socks[i+1] - socks[i]
        sys.stdout.write(str(total))
        return

    # s is odd.
    # We are allowed to leave one sock unpaired.
    # If we remove one sock at some index r (from the sorted list),
    # the optimal way to pair the remaining socks is to simply pair them as adjacent pairs.
    # We must choose r so that both left and right subarrays have even lengths.
    # It turns out that if s is odd, then the removal index MUST be even (0-indexed)
    # so that:
    #   - Left part (socks[0:r]) has r socks (and r is even).
    #   - Right part (socks[r+1:]) has s-r-1 socks, and since s is odd and r is even, s-r-1 is even.
    #
    # Precompute prefix pairing cost for socks[0 : r] (with r even).
    pre = [0] * (s + 1)
    # For an even-length prefix socks[0:i], the optimal pairing (adjacent pairing) cost is:
    # pre[i] = pre[i-2] + (socks[i-1] - socks[i-2])
    for i in range(2, s + 1, 2):
        pre[i] = pre[i-2] + (socks[i-1] - socks[i-2])

    # Precompute suffix pairing cost for segments of the form socks[i: s]
    # provided (s - i) is even.
    # We define suf[i] = cost to pair socks[i:] optimally (adjacent pairing),
    # if (s - i) is even.
    suf = [0] * (s + 2)  # extra space to avoid index error
    # We'll compute suf[i] for every i for which the segment length s-i is even.
    # If s-i >= 2, then:
    #   suf[i] = (socks[i+1] - socks[i]) + suf[i+2]
    # We iterate backwards.
    for i in range(s - 2, -1, -1):
        if ((s - i) & 1) == 0:
            suf[i] = (socks[i+1] - socks[i]) + suf[i+2]

    # Try all possible removals at an even index r.
    best = 10**18
    for r in range(0, s, 2):
        # Left part is socks[0:r] (r is even so pre[r] is defined).
        left_cost = pre[r]
        # Right part is socks[r+1: s]. Note: Since s is odd and r is even, (s - r - 1) is even.
        right_cost = suf[r+1] if (r + 1 <= s) else 0
        candidate = left_cost + right_cost
        if candidate < best:
            best = candidate

    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()