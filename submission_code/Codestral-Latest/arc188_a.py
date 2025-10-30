MOD = 998244353

def count_good_strings(S, K):
    N = len(S)
    dp = [[0] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 1

    for i in range(N):
        for j in range(i + 1):
            if S[i] == '?':
                for c in 'ABC':
                    dp[i + 1][j + (c in 'ABC')] = (dp[i + 1][j + (c in 'ABC')] + dp[i][j]) % MOD
            else:
                dp[i + 1][j + (S[i] in 'ABC')] = (dp[i + 1][j + (S[i] in 'ABC')] + dp[i][j]) % MOD

    result = 0
    for j in range(K, N + 1):
        result = (result + dp[N][j]) % MOD

    return result

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    S = data[2]

    print(count_good_strings(S, K))

if __name__ == "__main__":
    main()