class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        heights = sorted([(h, i) for i, h in enumerate(maximumHeight)], reverse=True)
        
        used_heights = set()
        result = 0
        
        for max_height, index in heights:
            for height in range(max_height, 0, -1):
                if height not in used_heights:
                    used_heights.add(height)
                    result += height
                    break
            else:
                return -1
        
        return result