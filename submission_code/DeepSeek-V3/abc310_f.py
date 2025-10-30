MOD = 998244353

def inv(x, mod):
    return pow(x, mod-2, mod)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize dp
    dp = [0] * 11
    dp[0] = 1
    
    for a in A:
        new_dp = [0] * 11
        for s in range(11):
            if dp[s]:
                for v in range(1, a+1):
                    if s + v <= 10:
                        new_dp[s + v] = (new_dp[s + v] + dp[s] * inv(a, MOD)) % MOD
        for s in range(11):
            dp[s] = (dp[s] + new_dp[s]) % MOD
    
    # Calculate the total probability
    total = 1
    for a in A:
        total = total * a % MOD
    
    # The probability is the sum of dp[1..10] divided by total
    prob = sum(dp[1:11]) % MOD
    prob = prob * inv(total, MOD) % MOD
    
    # The answer is prob * inv(1, MOD) % MOD, but since prob is already y / x, and x is not divisible by MOD, we can directly output prob
    print(prob)

if __name__ == "__main__":
    main()