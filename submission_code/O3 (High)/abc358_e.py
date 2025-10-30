import sys

MOD = 998244353

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    K = data[0]
    C = data[1:27]              # there are always 26 numbers

    # factorials and inverse factorials up to K
    fact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = fact[i - 1] * i % MOD

    inv_fact = [1] * (K + 1)
    inv_fact[K] = pow(fact[K], MOD - 2, MOD)
    for i in range(K, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD

    # polynomial q(x), q[deg] = coefficient of x^deg
    q = [1] + [0] * K           # start with 1

    for c in C:
        if c == 0:
            continue            # multiplying by 1, nothing changes
        c = min(c, K)           # we do not need exponents above K
        new_q = [0] * (K + 1)

        # convolution with P(x) = Σ_{t=0..c} x^t / t!
        for t in range(c + 1):
            coef = inv_fact[t]          # 1 / t!
            for j in range(K - t + 1):
                if q[j]:
                    new_q[j + t] = (new_q[j + t] + q[j] * coef) % MOD
        q = new_q                       # move to next letter

    # accumulate answer
    ans = 0
    for n in range(1, K + 1):
        ans = (ans + fact[n] * q[n]) % MOD

    print(ans)


if __name__ == "__main__":
    main()