MOD = 998244353

def main():
    import sys
    S = sys.stdin.read().strip()
    n = len(S)
    max_balance = n  # Maximum possible balance
    # Initialize previous DP
    previous_dp = [0] * (max_balance + 1)
    previous_dp[0] = 1
    for i in range(n):
        current_dp = [0] * (max_balance + 1)
        char = S[i]
        for j in range(max_balance + 1):
            if previous_dp[j]:
                if char == '(':
                    new_balance = j + 1
                    if new_balance <= max_balance:
                        current_dp[new_balance] = (current_dp[new_balance] + previous_dp[j]) % MOD
                elif char == ')':
                    new_balance = j - 1
                    if new_balance >= 0:
                        current_dp[new_balance] = (current_dp[new_balance] + previous_dp[j]) % MOD
                elif char == '?':
                    new_balance1 = j + 1
                    if new_balance1 <= max_balance:
                        current_dp[new_balance1] = (current_dp[new_balance1] + previous_dp[j]) % MOD
                    new_balance2 = j - 1
                    if new_balance2 >= 0:
                        current_dp[new_balance2] = (current_dp[new_balance2] + previous_dp[j]) % MOD
        previous_dp = current_dp
    print(previous_dp[0] % MOD)

if __name__ == "__main__":
    main()