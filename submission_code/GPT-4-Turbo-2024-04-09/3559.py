class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import defaultdict
        import sys
        
        # Create a dictionary to store all prefixes of each word
        prefix_dict = defaultdict(list)
        for word in words:
            for i in range(1, len(word) + 1):
                prefix_dict[word[:i]].append(word)
        
        # Dynamic programming array to store the minimum number of valid strings
        # dp[i] will be the minimum number of valid strings to form target[:i]
        dp = [sys.maxsize] * (len(target) + 1)
        dp[0] = 0  # Base case: 0 valid strings needed to form an empty target
        
        # Fill the dp array
        for i in range(1, len(target) + 1):
            for j in range(1, i + 1):
                prefix = target[j-1:i]
                if prefix in prefix_dict:
                    dp[i] = min(dp[i], dp[j-1] + 1)
        
        # If dp[len(target)] is still sys.maxsize, it means target cannot be formed
        return dp[len(target)] if dp[len(target)] != sys.maxsize else -1