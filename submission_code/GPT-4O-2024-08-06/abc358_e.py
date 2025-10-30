# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    K = int(data[0])
    C = list(map(int, data[1:]))
    
    # dp[length] will store the number of ways to form strings of length `length`
    dp = [0] * (K + 1)
    dp[0] = 1  # Base case: one way to form an empty string
    
    # Iterate over each letter
    for i in range(26):
        # Maximum occurrences of the current letter
        max_occurrences = C[i]
        
        # We need to update dp array considering this letter
        new_dp = [0] * (K + 1)
        
        # For each length, calculate the number of ways to form strings
        for length in range(K + 1):
            if dp[length] > 0:
                # Try adding 0 to max_occurrences of the current letter
                for add in range(max_occurrences + 1):
                    if length + add <= K:
                        new_dp[length + add] = (new_dp[length + add] + dp[length]) % MOD
        
        # Update dp with new_dp
        dp = new_dp
    
    # We need to sum up all dp[length] for 1 <= length <= K
    result = sum(dp[1:K + 1]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()