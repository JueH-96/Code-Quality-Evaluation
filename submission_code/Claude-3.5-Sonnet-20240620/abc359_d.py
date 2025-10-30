# YOUR CODE HERE
MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    dp = [0] * (1 << K)
    dp[0] = 1

    for i in range(N):
        new_dp = [0] * (1 << K)
        for mask in range(1 << K):
            if dp[mask] == 0:
                continue
            for c in ('A', 'B'):
                if S[i] != '?' and S[i] != c:
                    continue
                new_mask = ((mask << 1) | (1 if c == 'B' else 0)) & ((1 << K) - 1)
                if K <= i + 1:
                    substr = format(new_mask, f'0{K}b')
                    if is_palindrome(substr):
                        continue
                new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
        dp = new_dp

    return sum(dp) % MOD

N, K = map(int, input().split())
S = input().strip()

result = count_good_strings(N, K, S)
print(result)