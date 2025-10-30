class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        
        max_height = max(horizontalCut[0], m - horizontalCut[-1])
        max_width = max(verticalCut[0], n - verticalCut[-1])
        
        for i in range(1, len(horizontalCut)):
            max_height = max(max_height, horizontalCut[i] - horizontalCut[i-1])
        
        for i in range(1, len(verticalCut)):
            max_width = max(max_width, verticalCut[i] - verticalCut[i-1])
        
        return max_height * max_width