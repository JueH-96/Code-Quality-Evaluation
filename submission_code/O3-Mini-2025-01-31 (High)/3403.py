class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] = minimum number of balanced substrings partitioning s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for j in range(n):
            # frequency counts for substring s[i:j+1]
            freq = [0] * 26
            # Iterate backwards from index j to 0 to form every candidate substring ending at j
            for i in range(j, -1, -1):
                char_index = ord(s[i]) - ord('a')
                freq[char_index] += 1
                
                # Check if the substring s[i:j+1] is balanced.
                # A substring is balanced if all its nonzero frequency counts are equal.
                min_freq = float('inf')
                max_freq = 0
                for count in freq:
                    if count > 0:
                        if count < min_freq:
                            min_freq = count
                        if count > max_freq:
                            max_freq = count
                
                if min_freq == max_freq:
                    dp[j+1] = min(dp[j+1], dp[i] + 1)
                    
        return dp[n]

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Test Example 1
    s1 = "fabccddg"
    print("Input:", s1)
    print("Output:", sol.minimumSubstringsInPartition(s1))  # Expected Output: 3

    # Test Example 2
    s2 = "abababaccddb"
    print("Input:", s2)
    print("Output:", sol.minimumSubstringsInPartition(s2))  # Expected Output: 2