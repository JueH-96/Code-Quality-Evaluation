def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    lost = list(map(int, data[2:]))

    # freq[c] = how many socks remain of color c+1 (initially 2, reduced to 1 if lost)
    freq = [2] * N
    for c in lost:
        freq[c - 1] = 1  # each lost sock reduces that color's count from 2 to 1

    # Build the sorted list B of all remaining socks' colors
    # (Sorting is implicit by appending in ascending order of colors)
    B = []
    for i in range(N):
        color = i + 1
        if freq[i] == 2:
            B.append(color)
            B.append(color)
        elif freq[i] == 1:
            B.append(color)

    L = len(B)  # total number of remaining socks

    # If we have an even number of socks, they must all be paired
    # The pairing in ascending order has cost = sum(B[2i+1] - B[2i]).
    # That is the same as (sum of socks at odd positions) - (sum of socks at even positions).
    if L % 2 == 0:
        total = 0  # we will do one pass to sum odd indices minus even indices
        for i in range(0, L, 2):
            total -= B[i]
        for i in range(1, L, 2):
            total += B[i]
        print(total)
        return

    # Otherwise, we have an odd number of socks and must skip exactly one.
    # We still pair the remaining 2*(L//2) socks in consecutive order.
    # A known result is that if we let:
    #
    #   SEven[i] = sum of B[j] for j < i and j even
    #   SOdd[i]  = sum of B[j] for j < i and j odd
    #
    # and define
    #
    #   D[i] = SOdd[i] - SEven[i],
    #
    # then for skipping index r, the cost of pairing the new array is:
    #
    #   cost(r) = (sumEvenB - sumOddB) + D[r] + D[r+1]
    #
    # where sumEvenB = SEven[L], sumOddB = SOdd[L].
    #
    # So we compute these prefix sums, then compute cost(r) for r=0..L-1
    # and take the minimum.

    SEven = [0] * (L + 1)  # prefix sums of B's even indices
    SOdd = [0] * (L + 1)   # prefix sums of B's odd indices

    for i in range(L):
        SEven[i + 1] = SEven[i]
        SOdd[i + 1] = SOdd[i]
        if i % 2 == 0:
            SEven[i + 1] += B[i]
        else:
            SOdd[i + 1] += B[i]

    sumEvenB = SEven[L]
    sumOddB = SOdd[L]
    base = sumEvenB - sumOddB

    # Build D array: D[i] = SOdd[i] - SEven[i]
    D = [0] * (L + 1)
    for i in range(L + 1):
        D[i] = SOdd[i] - SEven[i]

    # Now find the minimal cost when skipping each possible index r
    ans = None
    for r in range(L):
        cost_r = base + D[r] + D[r + 1]
        if ans is None or cost_r < ans:
            ans = cost_r

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()