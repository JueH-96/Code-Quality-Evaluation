MOD = 998244353
max_n = 3000

fact = [1] * (max_n + 1)
for i in range(1, max_n + 1):
    fact[i] = fact[i-1] * i % MOD

inv_fact = [1] * (max_n + 1)
inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
for i in range(max_n - 1, -1, -1):
    inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

def comb(n, k):
    if k < 0 or k > n:
        return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N

    x = []
    valid = True
    for a in A:
        if a % 2 == 0:
            xi = a // 2
        else:
            xi = (a + 1) // 2
        if xi < 1 or xi > N:
            valid = False
        x.append(xi)
    if not valid or len(set(x)) != N:
        print(0)
        return

    u_set = set()
    for i in range(N):
        if B[i] == -1:
            continue
        b = B[i]
        if b % 2 == 0:
            zi = b // 2
        else:
            zi = (b + 1) // 2
        if zi < 1 or zi > N:
            valid = False
            break
        if zi == x[i]:
            valid = False
            break
        if zi in u_set:
            valid = False
            break
        u_set.add(zi)
    if not valid:
        print(0)
        return

    V = set(range(1, N+1)) - u_set
    t_indices = [i for i in range(N) if B[i] == -1]
    K = len(t_indices)
    C = 0
    for i in t_indices:
        if x[i] in V:
            C += 1

    ans = 0
    for k in range(0, C + 1):
        sign = pow(MOD - 1, k, MOD)
        c = comb(C, k)
        if (K - k) > max_n:
            fact_term = 0
        else:
            fact_term = fact[K - k]
        term = sign * c % MOD
        term = term * fact_term % MOD
        ans = (ans + term) % MOD
    print(ans % MOD)

if __name__ == '__main__':
    main()