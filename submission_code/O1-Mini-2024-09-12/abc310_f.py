# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    dp = [0] * 11
    dp[0] = 1
    x = 1
    for Ai in A:
        x = x * Ai % MOD
        new_dp = [0] * 11
        for s in range(0, 11):
            # Not including current die in the subset
            new_dp[s] = (new_dp[s] + dp[s] * Ai) % MOD
            # Including current die in the subset
            max_x = min(Ai, 10 - s)
            for xi in range(1, max_x + 1):
                new_dp[s + xi] = (new_dp[s + xi] + dp[s]) % MOD
        dp = new_dp
    y = dp[10]
    inv_x = pow(x, MOD-2, MOD)
    z = y * inv_x % MOD
    print(z)

if __name__ == "__main__":
    main()