import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    T = int(data[1])
    M = int(data[2])
    pairs = []
    idx = 3
    for _ in range(M):
        a = int(data[idx]) - 1
        b = int(data[idx+1]) - 1
        idx += 2
        pairs.append((a, b))
    full = (1 << N) - 1

    # Precompute which subsets are "good" (no incompatible pair inside)
    good = [True] * (1 << N)
    for S in range(1 << N):
        for (a, b) in pairs:
            if (S >> a) & 1 and (S >> b) & 1:
                good[S] = False
                break

    # dp[mask][k] = number of ways to partition 'mask' into k good blocks
    dp = [ [0] * (T+1) for _ in range(1 << N) ]
    dp[0][0] = 1

    for mask in range(1, full + 1):
        # find lowest set bit in mask to enforce canonical enumeration of submasks
        lowbit = mask & -mask
        # for each possible number of blocks k = 1..T
        for k in range(1, T+1):
            # iterate submasks of mask that include lowbit
            sub = mask
            while sub:
                if (sub & lowbit) and good[sub]:
                    dp[mask][k] += dp[mask ^ sub][k-1]
                sub = (sub - 1) & mask

    # answer: partition the full set into exactly T blocks
    print(dp[full][T])


if __name__ == "__main__":
    main()