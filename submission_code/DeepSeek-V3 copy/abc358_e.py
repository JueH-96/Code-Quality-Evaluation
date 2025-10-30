MOD = 998244353

def main():
    K = int(input())
    C = list(map(int, input().split()))
    
    # Initialize dp array
    # dp[l][c1][c2]...[c26] represents the number of strings of length l with counts c1, c2, ..., c26
    # Since the counts are bounded by C_i, we can limit the counts to min(C_i, K)
    
    # To simplify, we will use a dictionary to store the counts for each length
    # For each length l from 1 to K, we will compute the possible counts and sum the valid ones
    
    # Initialize the total count
    total = 0
    
    # We will use a dynamic programming approach where for each length, we iterate over all possible counts
    # and update the counts based on the previous length
    
    # Initialize the dp for length 0
    # For length 0, the only possible count is all zeros
    dp = {}
    dp[(0,) * 26] = 1
    
    for l in range(1, K+1):
        new_dp = {}
        for counts, cnt in dp.items():
            for i in range(26):
                if counts[i] < C[i]:
                    new_counts = list(counts)
                    new_counts[i] += 1
                    new_counts = tuple(new_counts)
                    if new_counts in new_dp:
                        new_dp[new_counts] = (new_dp[new_counts] + cnt) % MOD
                    else:
                        new_dp[new_counts] = cnt % MOD
        dp = new_dp
        # Sum all counts in dp for the current length
        for counts, cnt in dp.items():
            total = (total + cnt) % MOD
    
    print(total)

if __name__ == "__main__":
    main()