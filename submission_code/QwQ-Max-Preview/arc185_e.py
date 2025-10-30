MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    MAX_A = 10**5

    # Precompute smallest prime factors up to MAX_A
    max_spf = MAX_A
    spf = list(range(max_spf + 1))
    for i in range(2, int(max_spf**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_spf + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(x):
        factors = {}
        if x == 1:
            return factors
        while x != 1:
            p = spf[x]
            while x % p == 0:
                factors[p] = factors.get(p, 0) + 1
                x = x // p
        return factors

    def generate_divisors(factors):
        divisors = [1]
        for p, exp in factors.items():
            temp = []
            for d in divisors:
                current = d
                for e in range(exp + 1):
                    temp.append(current * (p ** e))
            divisors = temp
        return divisors

    def generate_divisors_with_mu(factors):
        divisors = [(1, 1)]
        for p, exp in factors.items():
            temp = []
            for d, mu in divisors:
                current_d = d
                current_mu = mu
                for e in range(exp + 1):
                    new_d = current_d * (p ** e)
                    if e > 1:
                        new_mu = 0
                    else:
                        if current_mu == 0:
                            new_mu = 0
                        else:
                            new_mu = current_mu * (-1) if e == 1 else current_mu
                    temp.append((new_d, new_mu))
            divisors = temp
        return divisors

    pow2 = [1] * (N + 1)
    for i in range(1, N + 1):
        pow2[i] = (pow2[i - 1] * 2) % MOD

    inv_2 = pow(2, MOD - 2, MOD)
    inv_2_pows = [1] * (N + 2)
    for i in range(1, N + 2):
        inv_2_pows[i] = (inv_2_pows[i - 1] * inv_2) % MOD

    frequency = [0] * (MAX_A + 2)
    prefix_T = 0
    output = []

    for j in range(1, N + 1):
        a = A[j - 1]
        factors_a = factorize(a)
        divisors_d = generate_divisors(factors_a)
        S_j = 0
        for d in divisors_d:
            y = a // d
            factors_y = factorize(y)
            divisors_k_mu = generate_divisors_with_mu(factors_y)
            contribution = 0
            for k, mu in divisors_k_mu:
                g = d * k
                sum_g = frequency[g]
                contribution += sum_g * mu
            contribution %= MOD
            S_j = (S_j + d * contribution) % MOD
        T_j = (S_j * inv_2_pows[j]) % MOD
        prefix_T = (prefix_T + T_j) % MOD
        ans = (pow2[j] * prefix_T) % MOD
        output.append(ans)
        # Update frequency for divisors of a
        divisors_aj = generate_divisors(factors_a)
        for g in divisors_aj:
            frequency[g] = (frequency[g] + pow2[j - 1]) % MOD

    print('
'.join(map(str, output)))

if __name__ == '__main__':
    main()