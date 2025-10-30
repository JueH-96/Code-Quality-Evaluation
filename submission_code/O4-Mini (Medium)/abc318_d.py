import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Read the upper-triangle weights and build a full symmetric matrix D.
    D = [[0]*N for _ in range(N)]
    for i in range(N-1):
        for j in range(i+1, N):
            w = int(next(it))
            D[i][j] = w
            D[j][i] = w

    # dp[mask] = maximum total weight of a matching on the set of vertices represented by 'mask'.
    # We allow leaving vertices unmatched (skip them).
    full = 1 << N
    dp = [0] * full

    # Iterate over masks in increasing order.
    for mask in range(1, full):
        # Find the least significant set bit: that's our "first" vertex to consider.
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        # Option 1: leave vertex i unmatched.
        best = dp[mask ^ (1 << i)]
        # Option 2: pair i with any other vertex j in 'mask'.
        rem = mask ^ (1 << i)
        # Iterate over all set bits j in rem.
        m = rem
        while m:
            lsb2 = m & -m
            j = lsb2.bit_length() - 1
            # Form the mask without i and j.
            without_ij = rem ^ (1 << j)
            val = dp[without_ij] + D[i][j]
            if val > best:
                best = val
            m ^= lsb2

        dp[mask] = best

    # The answer is the best matching on the full set of vertices.
    print(dp[full - 1])

if __name__ == "__main__":
    main()