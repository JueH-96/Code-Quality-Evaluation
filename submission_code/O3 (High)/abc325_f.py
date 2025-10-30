import sys
from math import ceil

def main() -> None:
    INF = 10 ** 18

    it = iter(sys.stdin.read().strip().split())
    N = int(next(it))
    D = [int(next(it)) for _ in range(N)]

    L1, C1, K1 = map(int, (next(it), next(it), next(it)))
    L2, C2, K2 = map(int, (next(it), next(it), next(it)))

    # Quick impossibility check: every single section must be coverable even
    # if we devote every sensor to that section.
    max_len_each = K1 * L1 + K2 * L2
    if any(d > max_len_each for d in D):
        print(-1)
        return

    # Use the sensor type that has the smaller upper bound as the DP dimension
    if K1 <= K2:
        LA, CA, KA = L1, C1, K1
        LB, CB, KB = L2, C2, K2
    else:
        LA, CA, KA = L2, C2, K2
        LB, CB, KB = L1, C1, K1

    # dp[a] = minimal number of type-B sensors needed
    # after processing some sections while having used `a` type-A sensors.
    dp = [INF] * (KA + 1)
    dp[0] = 0

    for d in D:
        t1_max = min(KA, (d + LA - 1) // LA)  # max type-A sensors that might help
        # pre-compute for the current section: type-B sensors needed
        need_b = [0] * (t1_max + 1)
        for a in range(t1_max + 1):
            remaining = d - a * LA
            if remaining <= 0:
                need_b[a] = 0
            else:
                need_b[a] = (remaining + LB - 1) // LB   # ceil division

        new_dp = [INF] * (KA + 1)
        for used_a in range(KA + 1):
            cur_b = dp[used_a]
            if cur_b == INF:
                continue
            # Try using `a_add` extra type-A sensors for this section
            for a_add in range(t1_max + 1):
                new_used_a = used_a + a_add
                if new_used_a > KA:
                    break
                v = cur_b + need_b[a_add]
                if v < new_dp[new_used_a]:
                    new_dp[new_used_a] = v
        dp = new_dp

    ans = INF
    for a in range(KA + 1):
        b = dp[a]
        if b <= KB:
            cost = a * CA + b * CB
            if cost < ans:
                ans = cost

    print(-1 if ans == INF else ans)

if __name__ == "__main__":
    main()