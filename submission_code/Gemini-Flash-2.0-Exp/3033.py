class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff_indices = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff_indices.append(i)
        
        if len(diff_indices) % 2 != 0:
            return -1
        
        dp = {}
        
        def solve(index):
            if index >= len(diff_indices):
                return 0
            
            if index in dp:
                return dp[index]
            
            option1 = x + solve(index + 2)
            option2 = 10**9
            if index + 1 < len(diff_indices):
                option2 = diff_indices[index+1] - diff_indices[index] + solve(index + 2)
            
            dp[index] = min(option1, option2)
            return dp[index]
        
        return solve(0)