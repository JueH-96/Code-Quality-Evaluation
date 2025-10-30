MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    K = int(data[0])
    C = list(map(int, data[1:27]))
    
    max_n = K
    # Precompute combination numbers C(n, k) mod MOD
    comb = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for n in range(max_n + 1):
        comb[n][0] = 1
        for k in range(1, n + 1):
            comb[n][k] = (comb[n-1][k-1] + comb[n-1][k]) % MOD
    
    # Initialize DP
    prev_dp = [0] * (K + 1)
    prev_dp[0] = 1  # Base case: one way to have a string of length 0 (empty string)
    
    for i in range(1, 27):
        c_i = C[i-1]
        curr_dp = [0] * (K + 1)
        for j in range(K + 1):
            max_k = min(c_i, j)
            for k in range(0, max_k + 1):
                curr_dp[j] = (curr_dp[j] + prev_dp[j - k] * comb[j][k]) % MOD
        prev_dp = curr_dp
    
    # Sum all valid strings from length 1 to K
    result = sum(prev_dp[1:K+1]) % MOD
    print(result)

if __name__ == "__main__":
    main()