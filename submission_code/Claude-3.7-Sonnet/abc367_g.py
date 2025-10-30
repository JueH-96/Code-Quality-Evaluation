from collections import defaultdict

def solve(N, M, K, A):
    MOD = 998244353
    
    # Initialize DP
    # dp[j][x] represents the number of subsequences of length j with XOR value x
    dp = [defaultdict(int) for _ in range(N + 1)]
    dp[0][0] = 1  # Empty subsequence has length 0 and XOR value 0
    
    for i, a in enumerate(A):
        for j in range(i, -1, -1):  # Iterate from largest length to smallest to avoid double counting
            for x, count in list(dp[j].items()):
                dp[j+1][x ^ a] = (dp[j+1][x ^ a] + count) % MOD
    
    result = 0
    for j in range(1, N + 1):
        if j % M == 0:  # Only consider lengths that are multiples of M
            for x, count in dp[j].items():
                score = pow(x, K, MOD) * count % MOD
                result = (result + score) % MOD
    
    return result

def main():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    print(solve(N, M, K, A))

if __name__ == "__main__":
    main()