class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                prefixes.add(word[:i])
        
        dp = {}
        def solve(index):
            if index == len(target):
                return 0
            if index in dp:
                return dp[index]
            
            ans = float('inf')
            for i in range(index + 1, len(target) + 1):
                prefix = target[index:i]
                if prefix in prefixes:
                    res = solve(i)
                    if res != float('inf'):
                        ans = min(ans, 1 + res)
            
            dp[index] = ans
            return ans
        
        res = solve(0)
        return res if res != float('inf') else -1