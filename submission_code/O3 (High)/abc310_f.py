import sys

MOD = 998244353          # given prime
MAX_SUM = 10             # target sum
BITS = MAX_SUM + 1       # we keep sums 0..10
FULL_MASK = (1 << BITS) - 1   # 11 bits -> 0..10


def main() -> None:
    # read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))

    # dp[mask] : probability (mod MOD) that after seeing current dice,
    # exactly the set of subset–sums contained in `mask` (bits 0..10) is reachable
    dp = [0] * (1 << BITS)
    dp[1] = 1                     # initially only sum 0 is reachable  -> bit 0 set

    for a in A:
        t = min(a, MAX_SUM)       # only values 1..10 can influence the tracked sums
        inv_a = pow(a, MOD - 2, MOD)   # modular inverse of the number of faces
        p_rest = (a - t) * inv_a % MOD # probability that the value is >10

        nd = [0] * (1 << BITS)

        # iterate over all current states
        for mask in range(1 << BITS):
            prob = dp[mask]
            if not prob:
                continue

            # case: value > 10  (no new sums can be formed)
            nd[mask] = (nd[mask] + prob * p_rest) % MOD

            # cases: value = 1 .. t
            for v in range(1, t + 1):
                new_mask = mask | ((mask << v) & FULL_MASK)
                nd[new_mask] = (nd[new_mask] + prob * inv_a) % MOD

        dp = nd                     # move to next die

    # sum probabilities of states where sum 10 is reachable (bit 10 is set)
    answer = 0
    TARGET_BIT = 1 << MAX_SUM
    for mask, prob in enumerate(dp):
        if mask & TARGET_BIT:
            answer += prob
    answer %= MOD

    print(answer)


if __name__ == "__main__":
    main()