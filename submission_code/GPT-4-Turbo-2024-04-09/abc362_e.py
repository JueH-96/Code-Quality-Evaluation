def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    MOD = 998244353
    
    # Result array to store the count of arithmetic subsequences of each length
    result = [0] * N
    
    # For length 1, every element is an arithmetic sequence
    result[0] = N
    
    # We use a dynamic programming approach to count the number of arithmetic subsequences
    # dp[i][d] will store the number of arithmetic subsequences ending at index i with common difference d
    from collections import defaultdict
    
    # List of dictionaries to store counts of arithmetic subsequences ending at each index with different common differences
    dp = [defaultdict(int) for _ in range(N)]
    
    for i in range(N):
        for j in range(i):
            # Calculate the common difference
            d = A[i] - A[j]
            
            # The number of subsequences ending at i with difference d is increased by the number of subsequences
            # ending at j with the same difference, plus the subsequence (A_j, A_i) itself
            count = dp[j][d] + 1
            
            # Update dp[i][d]
            dp[i][d] = (dp[i][d] + count) % MOD
            
            # Update the result for subsequences of length >= 2
            # We are adding subsequences of length k+2 where k is the length of subsequences counted in dp[j][d]
            result[1] = (result[1] + 1) % MOD  # This is for length 2 subsequences
            if count > 0:
                for k in range(2, N):
                    if k-2 < len(result):
                        result[k] = (result[k] + count) % MOD
    
    # Print the result array
    print(" ".join(map(str, result)))