def count_valid_strings(K, C):
    MOD = 998244353
    dp = [[0] * (K + 1) for _ in range(27)]
    dp[0][0] = 1
    
    for i in range(1, 27):
        for j in range(K + 1):
            for k in range(min(C[i-1] + 1, j + 1)):
                # Compute the combination: C(j, k)
                combination = 1
                for r in range(k):
                    combination = (combination * (j - r)) // (r + 1)
                
                dp[i][j] = (dp[i][j] + dp[i-1][j-k] * combination) % MOD
    
    result = sum(dp[26][j] for j in range(1, K + 1)) % MOD
    return result

def main():
    K = int(input())
    C = list(map(int, input().split()))
    
    result = count_valid_strings(K, C)
    print(result)

if __name__ == "__main__":
    main()