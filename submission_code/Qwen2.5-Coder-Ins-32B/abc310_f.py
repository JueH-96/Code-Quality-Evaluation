# YOUR CODE HERE
MOD = 998244353

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    
    total_outcomes = 1
    for a in A:
        total_outcomes = (total_outcomes * a) % MOD
    
    dp = [0] * 11
    dp[0] = 1
    
    for a in A:
        new_dp = [0] * 11
        for s in range(11):
            for v in range(1, a + 1):
                if s + v <= 10:
                    new_dp[s + v] = (new_dp[s + v] + dp[s]) % MOD
        dp = new_dp
    
    favorable_outcomes = dp[10]
    if favorable_outcomes == 0:
        print(0)
        return
    
    inverse_total_outcomes = mod_inverse(total_outcomes, MOD)
    result = (favorable_outcomes * inverse_total_outcomes) % MOD
    print(result)

solve()