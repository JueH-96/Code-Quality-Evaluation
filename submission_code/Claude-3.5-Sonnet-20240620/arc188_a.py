# YOUR CODE HERE
MOD = 998244353

def is_good_string(s):
    count = {'A': 0, 'B': 0, 'C': 0}
    for c in s:
        count[c] += 1
    return min(count.values()) > 0 or max(count.values()) > 1

def count_good_substrings(s):
    return sum(is_good_string(s[i:j]) for i in range(len(s)) for j in range(i+1, len(s)+1))

def solve(N, K, S):
    dp = [0] * (1 << N)
    dp[0] = 1

    for mask in range(1, 1 << N):
        for i in range(N):
            if mask & (1 << i):
                prev_mask = mask ^ (1 << i)
                for c in 'ABC':
                    if S[i] == '?' or S[i] == c:
                        new_s = ''.join([c if mask & (1 << j) else '?' for j in range(N)])
                        if count_good_substrings(new_s) >= K:
                            dp[mask] = (dp[mask] + dp[prev_mask]) % MOD

    return dp[(1 << N) - 1]

N, K = map(int, input().split())
S = input().strip()

result = solve(N, K, S)
print(result)