S = input().strip()
n = len(S)
mod = 998244353

if n % 2 != 0:
    print(0)
    exit()

# Initialize DP
current = [0] * (n + 2)
current[0] = 1

for i in range(n):
    next_dp = [0] * (n + 2)
    c = S[i]
    for j in range(len(current)):
        if current[j] == 0:
            continue
        if c == '(':
            new_j = j + 1
            if new_j <= n:
                next_dp[new_j] = (next_dp[new_j] + current[j]) % mod
        elif c == ')':
            new_j = j - 1
            if new_j >= 0:
                next_dp[new_j] = (next_dp[new_j] + current[j]) % mod
        else:
            # Replace with '('
            new_j1 = j + 1
            if new_j1 <= n:
                next_dp[new_j1] = (next_dp[new_j1] + current[j]) % mod
            # Replace with ')'
            new_j2 = j - 1
            if new_j2 >= 0:
                next_dp[new_j2] = (next_dp[new_j2] + current[j]) % mod
    current = next_dp

print(current[0] % mod)