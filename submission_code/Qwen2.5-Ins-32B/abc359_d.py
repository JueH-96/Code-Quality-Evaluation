import sys

def is_palindrome(sub):
    return sub == sub[::-1]

def count_good_strings(N, K, S):
    MOD = 998244353
    dp = [[0] * (1 << K) for _ in range(N + 1)]
    dp[0][0] = 1

    for i in range(N):
        for mask in range(1 << K):
            if dp[i][mask] == 0:
                continue
            for c in 'AB':
                if S[i] != '?' and S[i] != c:
                    continue
                new_mask = ((mask << 1) | (1 if c == 'B' else 0)) & ((1 << K) - 1)
                if not is_palindrome(''.join('B' if new_mask & (1 << j) else 'A' for j in range(K))):
                    dp[i + 1][new_mask] = (dp[i + 1][new_mask] + dp[i][mask]) % MOD

    return sum(dp[N]) % MOD

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    S = data[2]

    print(count_good_strings(N, K, S))