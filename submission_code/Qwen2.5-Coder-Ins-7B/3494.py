class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        
        total_cost = 0
        max_width = 0
        current_width = 0
        
        for cut in horizontalCut:
            total_cost += current_width * cut
            max_width = max(max_width, current_width)
            current_width = 0
        
        total_cost += max_width * n
        
        current_height = 0
        for cut in verticalCut:
            total_cost += current_height * cut
            current_height = 0
        
        total_cost += current_height * m
        
        return total_cost