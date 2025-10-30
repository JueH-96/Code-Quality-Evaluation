import sys

MOD = 998244353

# ------------------  combinatorics ------------------ #
def build_factorials(limit: int):
    fact = [1] * (limit + 1)
    for i in range(1, limit + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (limit + 1)
    inv_fact[limit] = pow(fact[limit], MOD - 2, MOD)
    for i in range(limit, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD
    # modular inverses of integers
    inv_int = [0] * (limit + 1)
    inv_int[1] = 1
    for i in range(2, limit + 1):
        inv_int[i] = MOD - MOD // i * inv_int[MOD % i] % MOD
    return fact, inv_fact, inv_int


def nCk(n: int, k: int, fact, inv_fact):
    if k < 0 or k > n:
        return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD


# number of completions with remaining length L and current outstanding r
def forest_count(L: int, r: int, fact, inv_fact, inv_int):
    if L == 0:
        return 1 if r == 0 else 0
    if r <= 0 or r > L:
        return 0
    s = 2 * L - r  # = 2L - r
    comb = nCk(s, L, fact, inv_fact)
    return r * comb % MOD * inv_int[s] % MOD


# ------------------  main ------------------ #
def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # pre–computations
    LIMIT = 2 * N           # we will need factorials up to 2N
    fact, inv_fact, inv_int = build_factorials(LIMIT)

    ans = 0
    outstanding = 1         # slots that still have to be filled
    broken = False

    for i in range(N):
        L = N - i - 1                   # remaining positions after i
        a = A[i]

        # range of v (< a) that keep the sequence still completable
        low_v = max(0, 2 - outstanding)             # guarantees r>=1 if L>0
        high_v = min(a - 1, L - outstanding + 1)    # guarantees r<=L
        if low_v <= high_v:
            c = outstanding - 1                     # r = c + v
            for v in range(low_v, high_v + 1):
                r = c + v
                ans = (ans + forest_count(L, r, fact, inv_fact, inv_int)) % MOD

        # follow the exact value of A[i] to keep equality with prefix
        outstanding = outstanding - 1 + a
        if i != N - 1 and outstanding <= 0:         # prefix already impossible
            broken = True
            break

    # if the whole sequence A itself is Polish, add it
    if not broken and outstanding == 0:
        ans = (ans + 1) % MOD

    print(ans % MOD)


if __name__ == "__main__":
    main()