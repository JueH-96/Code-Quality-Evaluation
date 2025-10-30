mod = 998244353
S = input().strip()
n = len(S)
if n % 2 != 0:
    print(0)
else:
    current = [0] * (n + 1)
    current[0] = 1
    for c in S:
        next_dp = [0] * (n + 1)
        for b in range(n + 1):
            if current[b] == 0:
                continue
            if c == '(':
                new_b = b + 1
                if new_b <= n:
                    next_dp[new_b] = (next_dp[new_b] + current[b]) % mod
            elif c == ')':
                new_b = b - 1
                if new_b >= 0:
                    next_dp[new_b] = (next_dp[new_b] + current[b]) % mod
            else:
                new_b = b + 1
                if new_b <= n:
                    next_dp[new_b] = (next_dp[new_b] + current[b]) % mod
                new_b = b - 1
                if new_b >= 0:
                    next_dp[new_b] = (next_dp[new_b] + current[b]) % mod
        current = next_dp
    print(current[0] % mod)