class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] represents minimum partitions needed for s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # empty string needs 0 partitions
        
        def is_balanced(substring):
            # Count frequency of each character
            freq = {}
            for char in substring:
                freq[char] = freq.get(char, 0) + 1
            
            # Check if all frequencies are the same
            frequencies = list(freq.values())
            return len(set(frequencies)) == 1
        
        # For each ending position
        for i in range(1, n + 1):
            # Try all possible starting positions
            for j in range(i):
                # Check if s[j:i] is balanced
                if is_balanced(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]