MOD = 998244353

def inv(x, mod):
    return pow(x, mod-2, mod)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize dp: dp[s] = number of ways to achieve sum s
    dp = [0] * 11
    dp[0] = 1
    
    for a in A:
        new_dp = [0] * 11
        for s in range(11):
            if dp[s]:
                for x in range(1, a+1):
                    if s + x < 11:
                        new_dp[s + x] = (new_dp[s + x] + dp[s]) % MOD
                new_dp[s] = (new_dp[s] + dp[s]) % MOD
        dp = new_dp
    
    # Total number of possible outcomes
    total = 1
    for a in A:
        total = (total * a) % MOD
    
    # Number of favorable outcomes is dp[10]
    favorable = dp[10]
    
    # Probability is favorable / total
    # Since MOD is prime and total is not divisible by MOD, we can compute the modular inverse
    inv_total = inv(total, MOD)
    probability = (favorable * inv_total) % MOD
    
    print(probability)

if __name__ == "__main__":
    main()