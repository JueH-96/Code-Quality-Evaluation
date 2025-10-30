import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    # Read the upper-triangular weights and build a full symmetric matrix
    w = [[0]*n for _ in range(n)]
    # For i from 0 to n-2, we read weights to j=i+1..n-1
    for i in range(n-1):
        for j in range(i+1, n):
            weight = int(next(it))
            w[i][j] = weight
            w[j][i] = weight

    # DP over subsets: dp[mask] = maximum total weight of a matching in the subgraph induced by mask
    # mask is a bitmask of selected vertices.
    NMASK = 1 << n
    dp = [0] * NMASK

    # Iterate over all masks in increasing order
    for mask in range(NMASK):
        # Count how many bits are on; if fewer than 2, no pairing possible
        # We can skip masks with fewer than two set bits
        if mask & (mask - 1) == 0:
            # This detects masks with 0 or 1 bits set
            continue

        # Find the lowest-numbered set bit i
        # Using bit & -bit to isolate lowest bit, then bit_length-1 to get the index
        lowbit = mask & -mask
        i = (lowbit.bit_length() - 1)

        # Try pairing i with any higher j in mask
        rest_after_i = mask ^ lowbit
        subbest = 0
        m = rest_after_i
        while m:
            # isolate lowest bit of m, get its index j
            lb = m & -m
            j = lb.bit_length() - 1
            # form the mask with i and j removed
            prev_mask = mask ^ (1 << i) ^ (1 << j)
            # candidate matching sum
            val = dp[prev_mask] + w[i][j]
            if val > subbest:
                subbest = val
            # clear the bit and continue
            m ^= lb

        dp[mask] = subbest

    # The answer is dp for the full set of vertices
    full_mask = (1 << n) - 1
    print(dp[full_mask])


if __name__ == "__main__":
    main()