import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Precompute sum of stones for every subset mask
    ALL = 1 << N
    sum_mask = [0] * ALL
    for mask in range(1, ALL):
        # take lowest set bit
        lsb = mask & -mask
        bindex = (lsb.bit_length() - 1)
        sum_mask[mask] = sum_mask[mask ^ lsb] + A[bindex]

    # dp[mask] = set of possible XOR values for partitioning mask
    dp = [set() for _ in range(ALL)]
    dp[0].add(0)

    # We process masks in increasing popcount so that dp[rest] is ready
    masks_by_pop = [[] for _ in range(N+1)]
    for mask in range(ALL):
        masks_by_pop[mask.bit_count()].append(mask)

    for size in range(1, N+1):
        for mask in masks_by_pop[size]:
            # find lowest bit in mask to anchor our block choice
            bbit = mask & -mask
            # iterate all non-empty submasks that include bbit
            sub = mask
            # We'll build dp[mask] by taking a block=sub and merging with dp[mask^sub]
            # Use a local reference for speed
            dp_mask = dp[mask]
            dp_rest = dp  # alias
            sum_m = sum_mask
            while sub:
                if (sub & bbit):
                    rest = mask ^ sub
                    ssub = sum_m[sub]
                    # merge
                    for x in dp_rest[rest]:
                        dp_mask.add(x ^ ssub)
                sub = (sub - 1) & mask

    result = len(dp[ALL-1])
    print(result)

if __name__ == "__main__":
    main()