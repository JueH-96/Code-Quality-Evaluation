class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        heights = [0] * n
        
        # Sort the maximumHeight array in descending order
        maximumHeight.sort(reverse=True)
        
        # Assign heights to the towers
        for i in range(n):
            for h in range(1, maximumHeight[i] + 1):
                if h not in heights:
                    heights[i] = h
                    break
            else:
                return -1
        
        return sum(heights)