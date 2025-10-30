import sys

MOD = 998244353
INV2 = (MOD + 1) // 2          # 2^{-1}  modulo MOD


def prepare_fact(n):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD

    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], MOD - 2, MOD)
    for i in range(n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD
    return fact, inv_fact


def main() -> None:
    it = map(int, sys.stdin.read().strip().split())
    N = next(it)
    A = [next(it) for _ in range(N)]

    # build run lengths
    runs = []
    cur = A[0]
    cnt = 1
    for v in A[1:]:
        if v == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur = v
            cnt = 1
    runs.append(cnt)

    # every run must be odd and the first value must be 1
    if A[0] != 1 or any(l % 2 == 0 for l in runs):
        print(0)
        return

    # t_j,  K = Σ t_j
    t_list = [(l - 1) // 2 for l in runs]
    K = sum(t_list)

    # factorials up to N are enough
    fact, inv_fact = prepare_fact(N)

    ans = fact[K]                         # K!
    for t in t_list:
        if t == 0:
            continue
        # multiply by   (2t)! * 2^{-t} * (t!)^{-2}
        ans = ans * fact[2 * t] % MOD
        ans = ans * pow(INV2, t, MOD) % MOD
        inv_t = inv_fact[t]
        ans = ans * inv_t % MOD
        ans = ans * inv_t % MOD

    print(ans % MOD)


if __name__ == "__main__":
    main()