class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        
        # If we need n different heights, we need at least numbers from 1 to n
        min_required = n * (n + 1) // 2
        
        # Check if we can assign n different heights
        # Sort in descending order to try to maximize sum
        maximumHeight.sort(reverse=True)
        
        # If the largest possible height can't accommodate the nth number
        # we need, then it's impossible
        for i in range(n):
            if maximumHeight[i] < n - i:
                return -1
                
        # Now we know it's possible, let's maximize the sum
        # Start with minimum required numbers
        heights = list(range(1, n + 1))
        
        # Try to increase each number as much as possible from largest to smallest
        # while staying within maximumHeight constraints
        for i in range(n-1, -1, -1):
            heights[i] = min(maximumHeight[i], maximumHeight[i])
            
            # Make sure we don't exceed any previous number
            if i > 0:
                heights[i] = min(heights[i], heights[i-1] - 1)
                
        return sum(heights)