import sys

MOD = 998244353

def main():
    K = int(sys.stdin.readline())
    C = list(map(int, sys.stdin.readline().split()))

    max_n = K
    # Precompute factorial and inverse factorial arrays
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n -1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    # Initialize coefficient array
    coeff = [0] * (K + 1)
    coeff[0] = 1

    for c in C:
        new_coeff = [0] * (K + 1)
        for new_m in range(K + 1):
            total = 0
            max_k = min(c, new_m)
            for k in range(0, max_k + 1):
                m_prev = new_m - k
                total += coeff[m_prev] * inv_fact[k]
            new_coeff[new_m] = total % MOD
        coeff = new_coeff

    # Calculate the final answer
    ans = 0
    for m in range(1, K + 1):
        ans = (ans + coeff[m] * fact[m]) % MOD
    print(ans)

if __name__ == "__main__":
    main()