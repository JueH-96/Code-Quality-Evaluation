class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        
        for i in range(1, n):
            if s[i] != s[i-1]:
                # Choose minimum between flipping left side or right side
                cost += min(i, n-i)
                
        return cost