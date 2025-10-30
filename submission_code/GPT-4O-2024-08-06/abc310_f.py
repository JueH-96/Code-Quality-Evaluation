# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # dp[i][s] will be the number of ways to get sum s using the first i dice
    dp = [[0] * 11 for _ in range(N + 1)]
    dp[0][0] = 1  # One way to get sum 0 with 0 dice
    
    for i in range(1, N + 1):
        for s in range(11):
            for k in range(1, A[i - 1] + 1):
                if s + k <= 10:
                    dp[i][s + k] = (dp[i][s + k] + dp[i - 1][s]) % MOD
    
    # Total number of outcomes
    total_outcomes = 1
    for a in A:
        total_outcomes = (total_outcomes * a) % MOD
    
    # Number of valid outcomes
    valid_outcomes = dp[N][10]
    
    # Probability as a fraction valid_outcomes / total_outcomes
    # We need to compute this modulo 998244353
    # valid_outcomes * total_outcomes^(-1) % MOD
    # Using Fermat's Little Theorem to find the modular inverse
    def mod_inverse(x, mod):
        return pow(x, mod - 2, mod)
    
    probability_mod = (valid_outcomes * mod_inverse(total_outcomes, MOD)) % MOD
    
    print(probability_mod)

main()