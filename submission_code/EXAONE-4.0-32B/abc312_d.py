mod = 998244353

def main():
    s = input().strip()
    n = len(s)
    dp = [0] * (n + 2)
    dp[0] = 1
    
    for c in s:
        new_dp = [0] * (n + 2)
        for j in range(n + 1):
            if dp[j] == 0:
                continue
            if c == '(' or c == '?':
                new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % mod
            if c == ')' or c == '?':
                if j > 0:
                    new_dp[j - 1] = (new_dp[j - 1] + dp[j]) % mod
        dp = new_dp
    
    print(dp[0] % mod)

if __name__ == "__main__":
    main()