class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        def is_balanced(sub):
            if not sub:
                return True
            counts = {}
            for char in sub:
                counts[char] = counts.get(char, 0) + 1
            
            if not counts:
                return True
            
            first_count = list(counts.values())[0]
            for count in counts.values():
                if count != first_count:
                    return False
            return True

        
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]