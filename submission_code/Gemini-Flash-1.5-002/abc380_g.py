MOD = 998244353

def inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def combinations(n, k, fact, inv_fact):
    if k < 0 or k > n:
        return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    fact = [1] * (n + 1)
    inv_fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] * i % MOD
        inv_fact[i] = inverse(fact[i], MOD)

    total_inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                total_inversions += 1

    expected_inversions = 0
    for i in range(n - k + 1):
        sub_array = p[i:i+k]
        
        inversions_sub = 0
        for j in range(k):
            for l in range(j+1,k):
                if sub_array[j] > sub_array[l]:
                    inversions_sub +=1

        
        expected_inversions_sub = (k * (k-1) // 2) * inverse(2,MOD) % MOD

        
        expected_inversions = (expected_inversions + total_inversions - inversions_sub + expected_inversions_sub) % MOD

    print((expected_inversions * inverse(n - k + 1, MOD)) % MOD)

solve()