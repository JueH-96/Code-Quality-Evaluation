class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best_sum = 0
        
        for peak in range(n):
            h = [0] * n
            h[peak] = maxHeights[peak]
            
            for j in range(peak - 1, -1, -1):
                h[j] = min(maxHeights[j], h[j + 1])
            
            for j in range(peak + 1, n):
                h[j] = min(maxHeights[j], h[j - 1])
            
            current_sum = sum(h)
            if current_sum > best_sum:
                best_sum = current_sum
                
        return best_sum