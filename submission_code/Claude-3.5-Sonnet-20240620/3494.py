class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in ascending order
        horizontalCut.sort()
        verticalCut.sort()
        
        # Add the boundaries
        horizontalCut = [0] + horizontalCut + [m]
        verticalCut = [0] + verticalCut + [n]
        
        # Calculate the maximum gap between cuts
        max_horizontal = max(horizontalCut[i+1] - horizontalCut[i] for i in range(len(horizontalCut) - 1))
        max_vertical = max(verticalCut[i+1] - verticalCut[i] for i in range(len(verticalCut) - 1))
        
        # Calculate the total cost
        total_cost = 0
        for h in horizontalCut[1:-1]:
            total_cost += max_vertical * h
        for v in verticalCut[1:-1]:
            total_cost += max_horizontal * v
        
        return total_cost