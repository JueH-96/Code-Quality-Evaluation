class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        res = 0
        
        for i in range(n):
            total = 0
            peak = maxHeights[i]
            total += peak
            
            last = peak
            for j in range(i-1, -1, -1):
                last = min(maxHeights[j], last)
                total += last
                
            last = peak
            for j in range(i+1, n):
                last = min(maxHeights[j], last)
                total += last
                
            if total > res:
                res = total
                
        return res