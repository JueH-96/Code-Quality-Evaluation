class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Create a set of valid prefixes from the words
        valid_prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                valid_prefixes.add(word[:i])
        
        # Initialize the dp array where dp[i] is the minimum number of valid strings to form target[:i]
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0  # Base case: 0 valid strings needed to form an empty target
        
        # Fill the dp array
        for i in range(1, len(target) + 1):
            for j in range(i):
                if target[j:i] in valid_prefixes:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        # If dp[len(target)] is still inf, it means we cannot form the target
        return dp[len(target)] if dp[len(target)] != float('inf') else -1