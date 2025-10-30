def solve():
    n = int(input())
    a_values = list(map(int, input().split()))
    
    MOD = 998244353
    max_target = 10
    
    # dp[i][j] = probability that there's a subset of the first i dice that sums to j
    dp = [[0 for _ in range(max_target + 1)] for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        a_i = a_values[i - 1]
        for j in range(max_target + 1):
            sum_prob = 0
            for v in range(1, a_i + 1):
                if j >= v:
                    # P(i, j, v) is the probability that, after rolling the first i dice and die i shows value v,
                    # there's a subset of the dice such that the sum of their values is exactly j.
                    p_i_j_v = 1 - (1 - dp[i-1][j]) * (1 - dp[i-1][j-v])
                else:
                    p_i_j_v = dp[i-1][j]
                sum_prob += p_i_j_v
            dp[i][j] = sum_prob / a_i
    
    # The answer is the probability that there's a subset of the dice that sums to max_target
    probability = dp[n][max_target]
    
    # Convert the floating-point probability to a fraction
    from fractions import Fraction
    frac = Fraction(probability).limit_denominator()
    p, q = frac.numerator, frac.denominator
    
    # Find z such that q * z ≡ p (mod MOD)
    def mod_inverse(a, m):
        g, x, y = extended_gcd(a % m, m)
        if g != 1:
            raise Exception('Modular inverse does not exist')
        else:
            return (x % m + m) % m
    
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        else:
            g, x, y = extended_gcd(b % a, a)
            return g, y - (b // a) * x, x
    
    z = (p * mod_inverse(q, MOD)) % MOD
    return z

print(solve())