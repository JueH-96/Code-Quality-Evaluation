import sys
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    max_A = max(A) if A else 0

    # Precompute smallest prime factors (SPF) up to max_A
    max_spf = max_A if max_A >= 2 else 2
    SPF = list(range(max_spf + 1))
    for i in range(2, int(max_spf**0.5) + 1):
        if SPF[i] == i:
            for j in range(i*i, max_spf+1, i):
                if SPF[j] == j:
                    SPF[j] = i

    # Function to get all divisors of n
    def get_divisors(n):
        if n == 0:
            return []
        factors = {}
        while n > 1:
            p = SPF[n]
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n = n // p
        divisors = [1]
        for p, exp in factors.items():
            temp = []
            for d in divisors:
                current = d
                for e in range(1, exp + 1):
                    current *= p
                    temp.append(current)
            divisors += temp
        return divisors

    # Precompute powers of 2 and their modular inverses
    max_pow = N
    pow_2 = [1] * (max_pow + 1)
    for i in range(1, max_pow + 1):
        pow_2[i] = (pow_2[i-1] * 2) % MOD
    inv_2 = pow(2, MOD-2, MOD)
    inv_2_pows = [1] * (max_pow + 1)
    for i in range(1, max_pow + 1):
        inv_2_pows[i] = (inv_2_pows[i-1] * inv_2) % MOD

    # Main processing
    from collections import defaultdict
    divisor_sums = defaultdict(int)
    prefix_sum = 0

    for j in range(N):
        a_j = A[j]
        D = get_divisors(a_j) if a_j != 0 else []
        D = list(set(D))
        sorted_D = sorted(D, reverse=True)
        sum_multiples = defaultdict(int)
        E = {}

        for d in sorted_D:
            S_d = divisor_sums[d] % MOD
            e = (S_d - sum_multiples.get(d, 0)) % MOD
            E[d] = e
            d_divisors = get_divisors(d)
            d_divisors = [x for x in d_divisors if x in D]
            for d_prime in d_divisors:
                sum_multiples[d_prime] = (sum_multiples[d_prime] + e) % MOD

        sum_Tj = 0
        for d in sorted_D:
            sum_Tj = (sum_Tj + d * E[d]) % MOD
        inv_2j = inv_2_pows[j+1]
        Tj = sum_Tj * inv_2j % MOD
        prefix_sum = (prefix_sum + Tj) % MOD

        current_pow = pow_2[j+1]
        for d in D:
            divisor_sums[d] = (divisor_sums[d] + current_pow) % MOD

        ans = prefix_sum * pow_2[j] % MOD
        print(ans)

if __name__ == '__main__':
    main()