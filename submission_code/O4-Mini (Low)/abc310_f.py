import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We track dp over bitmasks of reachable sums 0..9 (bit 10 is failure -> we exclude those states).
    # dp[mask] = probability (mod MOD) of having reachable-set = mask (no way to reach sum 10 yet).
    # Initialize: only sum 0 is reachable.
    dp = [0] * (1<<10)
    dp[1] = 1  # mask=1<<0

    for Ai in A:
        invAi = pow(Ai, MOD-2, MOD)
        newdp = [0] * (1<<10)
        # For each state mask without bit10:
        for mask in range(1<<10):
            prob = dp[mask]
            if prob == 0:
                continue
            # Build forbidden shifts that would create sum 10:
            # if s in reachable (mask>>s &1), then t = 10 - s is forbidden (1<=t<=Ai)
            forbidden = [False] * 11  # we only need up to 10
            for s in range(10):
                if (mask >> s) & 1:
                    t = 10 - s
                    if 1 <= t <= Ai:
                        forbidden[t] = True

            # Handle small shifts t = 1..min(9,Ai)
            up = min(9, Ai)
            for t in range(1, up+1):
                if forbidden[t]:
                    continue
                # compute new mask: shift existing bits by t, union, but only <=9:
                m2 = mask
                # shift bits:
                shifted = (mask << t) & ((1<<10)-1)
                m2 |= shifted
                # add probability mass:
                newdp[m2] = (newdp[m2] + prob * invAi) % MOD

            # Handle large shifts t >= 10:
            # total large = Ai - 9 if Ai>9 else 0
            total_large = Ai - 9 if Ai > 9 else 0
            if total_large > 0:
                # is t=10 forbidden?
                forbid10 = 1 if Ai >= 10 and forbidden[10] else 0
                large_count = total_large - forbid10
                if large_count > 0:
                    # for any t>=11 (or 10 if not forbidden), shifting by t puts all bits >9,
                    # so reachable bits don't expand (remain mask)
                    add = prob * large_count % MOD * invAi % MOD
                    newdp[mask] = (newdp[mask] + add) % MOD

        dp = newdp

    # sum(dp[mask]) over masks that never reached sum 10: that's prob_bad
    prob_bad = sum(dp) % MOD
    # desired probability = 1 - prob_bad mod
    ans = (1 - prob_bad) % MOD
    print(ans)

if __name__ == "__main__":
    main()