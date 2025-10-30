import sys

MOD = 998244353


def make_primes(limit: int):
    """sieve"""
    is_prime = [True] * (limit + 1)
    primes = []
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
            step = p
            start = p * p
            for q in range(start, limit + 1, step):
                is_prime[q] = False
    return primes


PRIMES = make_primes(1000)  # all primes we ever need ( 2 … 997 )


def factorise(x: int):
    """prime factorisation for  x ( 1 ≤ x ≤ 1000 )"""
    res = {}
    for p in PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            res[p] = cnt
    if x > 1:  # x itself is a prime (the big one, ≤ 1000)
        res[x] = 1
    return res


def contribution_of_prime(p: int, diffs):
    """
    diffs : list of exponent differences  ( |s_i - s_{i+1}| = diffs[i] )
    returns   Σ  p^{ Σ s_i }   over all sequences  (s_1 … s_N)
              with the above consecutive absolute‐difference constraint
              and   min(s_i) = 0
    """
    total = sum(diffs)                  # Maximal exponent ever reached
    if total == 0:                      # everything forced to 0
        return 1

    # p-powers up to `total`
    p_pow = [1] * (total + 1)
    for i in range(1, total + 1):
        p_pow[i] = (p_pow[i - 1] * p) % MOD

    size = total + 1                    # exponents range 0 … total

    # dp_no[v]   : weight of sequences ending with exponent v,
    #              no zero visited yet  ( min > 0 )
    # dp_yes[v]  : weight of sequences ending with exponent v,
    #              zero already visited ( min == 0 )
    dp_no = [0] * size
    dp_yes = [0] * size
    dp_yes[0] = 1                       # starting with s1 = 0
    for v in range(1, size):            # starting with s1 = v  (v>0)
        dp_no[v] = p_pow[v]

    for d in diffs:
        nxt_no = [0] * size
        nxt_yes = [0] * size
        if d == 0:
            for v in range(size):
                w = dp_no[v]
                if w:
                    val = (w * p_pow[v]) % MOD          # v' = v
                    nxt_no[v] = (nxt_no[v] + val) % MOD
                w = dp_yes[v]
                if w:
                    val = (w * p_pow[v]) % MOD
                    nxt_yes[v] = (nxt_yes[v] + val) % MOD
        else:
            for v in range(size):
                # transitions coming from states whose last exponent is `v`
                if dp_no[v]:
                    w = dp_no[v]
                    vp = v + d
                    if vp <= total:
                        nxt_no[vp] = (nxt_no[vp] + w * p_pow[vp]) % MOD
                    if v >= d:
                        vm = v - d
                        val = (w * p_pow[vm]) % MOD
                        if vm == 0:
                            nxt_yes[0] = (nxt_yes[0] + val) % MOD
                        else:
                            nxt_no[vm] = (nxt_no[vm] + val) % MOD
                if dp_yes[v]:
                    w = dp_yes[v]
                    vp = v + d
                    if vp <= total:
                        nxt_yes[vp] = (nxt_yes[vp] + w * p_pow[vp]) % MOD
                    if v >= d:
                        vm = v - d
                        nxt_yes[vm] = (nxt_yes[vm] + w * p_pow[vm]) % MOD
        dp_no, dp_yes = nxt_no, nxt_yes

    return sum(dp_yes) % MOD


def main() -> None:
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Collect exponent lists for every prime that appears in any A_i
    prime_to_diffs = {}                 # p -> list of length N-1 (all zeros)
    for idx, x in enumerate(A):
        for p, cnt in factorise(x).items():
            if p not in prime_to_diffs:
                prime_to_diffs[p] = [0] * (N - 1)
            prime_to_diffs[p][idx] = cnt
        # any primes not dividing this A_i just stay at 0

    answer = 1
    for p, diffs in prime_to_diffs.items():
        contrib = contribution_of_prime(p, diffs)
        answer = (answer * contrib) % MOD

        if answer == 0:                 # early stop if product already zero
            break

    print(answer % MOD)


if __name__ == "__main__":
    main()