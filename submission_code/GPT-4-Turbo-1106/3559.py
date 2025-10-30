class Solution:
    def minValidStrings(self, words: list[str], target: str) -> int:
        # Create a list of prefixes for each word
        prefixes = {word[:i] for word in words for i in range(1, len(word) + 1)}
        
        # Initialize the dp array with infinities, and set dp[0] to 0
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0
        
        # Fill the dp array
        for i in range(1, len(target) + 1):
            for j in range(i):
                if target[j:i] in prefixes:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        # If dp[-1] is still infinity, it means target can't be formed
        return -1 if dp[-1] == float('inf') else dp[-1]