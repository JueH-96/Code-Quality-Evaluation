def count_lcs_strings(N, M, S):
    MOD = 998244353
    total_strings = pow(26, M, MOD)
    
    # dp[k][j] will store the number of strings of length j with LCS of length k with S
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    
    # Base case: LCS of length 0 with any string is 1 (the empty string)
    for j in range(M + 1):
        dp[0][j] = 1
    
    for k in range(1, N + 1):
        for j in range(1, M + 1):
            # If we do not use the k-th character of S
            dp[k][j] = dp[k][j - 1] * 26 % MOD
            
            # If we use the k-th character of S
            dp[k][j] += dp[k - 1][j - 1] % MOD
            dp[k][j] %= MOD
            
            # If we match the k-th character of S with the j-th character of the string
            if j > 0:
                dp[k][j] += dp[k][j - 1] % MOD
                dp[k][j] %= MOD
    
    # Calculate the answers for each k
    answers = [0] * (N + 1)
    for k in range(N + 1):
        answers[k] = (total_strings - dp[k][M]) % MOD
    
    return answers

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    answers = count_lcs_strings(N, M, S)
    
    print(' '.join(map(str, answers)))

if __name__ == "__main__":
    main()