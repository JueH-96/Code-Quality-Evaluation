class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort(reverse=True)
        n = len(usageLimits)
        
        result = 0
        minVal = float('inf')
        
        for i in range(n):
            minVal = min(minVal, usageLimits[i] + i)
            if minVal >= i + 1:
                result = i + 1
            else:
                break
        
        return result