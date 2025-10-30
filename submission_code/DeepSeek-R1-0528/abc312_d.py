mod = 998244353

def main():
    s = input().strip()
    n = len(s)
    if n % 2 != 0:
        print(0)
        return
    
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for i in range(n):
        next_dp = [0] * (n + 1)
        for j in range(i + 1):
            if dp[j] == 0:
                continue
            if s[i] == '(':
                if j + 1 <= n:
                    next_dp[j + 1] = (next_dp[j + 1] + dp[j]) % mod
            elif s[i] == ')':
                if j > 0:
                    next_dp[j - 1] = (next_dp[j - 1] + dp[j]) % mod
            else:
                if j + 1 <= n:
                    next_dp[j + 1] = (next_dp[j + 1] + dp[j]) % mod
                if j > 0:
                    next_dp[j - 1] = (next_dp[j - 1] + dp[j]) % mod
        dp = next_dp
    
    print(dp[0] % mod)

if __name__ == "__main__":
    main()