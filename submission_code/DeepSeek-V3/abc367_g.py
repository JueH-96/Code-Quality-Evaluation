MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    # Precompute the number of subsequences of length l for each l
    # We need to count the number of subsequences of length l where l is a multiple of M
    # The total number of subsequences of length l is C(N, l)
    # We need to sum over all l that are multiples of M, and for each such l, compute the sum of (xor of the subsequence)^K
    
    # Since N can be up to 2e5, we need an efficient way to handle this
    # We can use dynamic programming to count the number of subsequences of each length and their XOR sums
    
    # Initialize dp[l][x] as the number of subsequences of length l with XOR sum x
    # We will iterate over each element in A and update the dp table
    
    # Initialize dp[0][0] = 1
    # For each element a in A, update the dp table
    # For each l from 0 to N-1, and for each x in dp[l], update dp[l+1][x ^ a] += dp[l][x]
    
    # Since the XOR sum can be up to 2^20, we can manage the size of x
    
    # Initialize dp as a list of dictionaries
    dp = [dict() for _ in range(N+1)]
    dp[0][0] = 1
    
    for a in A:
        new_dp = [dict() for _ in range(N+1)]
        for l in range(N):
            for x, cnt in dp[l].items():
                # Option 1: not take a
                if x in new_dp[l]:
                    new_dp[l][x] = (new_dp[l][x] + cnt) % MOD
                else:
                    new_dp[l][x] = cnt % MOD
                # Option 2: take a
                new_x = x ^ a
                if new_x in new_dp[l+1]:
                    new_dp[l+1][new_x] = (new_dp[l+1][new_x] + cnt) % MOD
                else:
                    new_dp[l+1][new_x] = cnt % MOD
        dp = new_dp
    
    # Now, for each l that is a multiple of M, sum over all x in dp[l] the value (x^K) * cnt
    total = 0
    for l in range(1, N+1):
        if l % M == 0:
            for x, cnt in dp[l].items():
                total = (total + pow(x, K, MOD) * cnt) % MOD
    print(total)

if __name__ == "__main__":
    main()