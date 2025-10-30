class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for peak in range(n):
            curr_sum = maxHeights[peak]
            prev = maxHeights[peak]
            
            # Go left from peak
            for i in range(peak-1, -1, -1):
                prev = min(prev, maxHeights[i])
                curr_sum += prev
            
            # Go right from peak
            prev = maxHeights[peak]
            for i in range(peak+1, n):
                prev = min(prev, maxHeights[i])
                curr_sum += prev
                
            max_sum = max(max_sum, curr_sum)
            
        return max_sum