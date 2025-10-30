MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    K = int(data[0])
    C = list(map(int, data[1:27]))
    
    # Initialize dp array
    # dp[l][c1][c2]...[c26] is not feasible due to high dimensions
    # Instead, use a dictionary or a list of dictionaries to represent the counts
    
    # We will use a dictionary where the key is a tuple representing the counts of each letter
    # and the value is the number of ways to achieve that count
    
    # Initialize with empty string
    dp = {tuple([0]*26): 1}
    
    result = 0
    
    for _ in range(K):
        new_dp = {}
        for counts, ways in dp.items():
            # Add each possible letter
            for i in range(26):
                if counts[i] < C[i]:
                    new_counts = list(counts)
                    new_counts[i] += 1
                    new_counts_tuple = tuple(new_counts)
                    if new_counts_tuple in new_dp:
                        new_dp[new_counts_tuple] = (new_dp[new_counts_tuple] + ways) % MOD
                    else:
                        new_dp[new_counts_tuple] = ways % MOD
        # Update dp
        dp = new_dp
        # Sum all ways in dp to get the total for the current length
        total = sum(dp.values()) % MOD
        result = (result + total) % MOD
    
    print(result)

if __name__ == "__main__":
    main()