class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut = [0] + sorted(horizontalCut) + [m]
        verticalCut = [0] + sorted(verticalCut) + [n]
        h_cost = 0
        v_cost = 0
        
        for i in range(1, len(horizontalCut)):
            h_cost += horizontalCut[i] - horizontalCut[i-1]
        
        for j in range(1, len(verticalCut)):
            v_cost += verticalCut[j] - verticalCut[j-1]
        
        return h_cost + v_cost