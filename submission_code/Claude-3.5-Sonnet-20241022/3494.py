class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in ascending order
        horizontalCut.sort()
        verticalCut.sort()
        
        # Add boundary points
        h_cuts = [0] + horizontalCut + [m]
        v_cuts = [0] + verticalCut + [n]
        
        # Get the maximum distance between consecutive cuts
        max_h = max(h_cuts[i+1] - h_cuts[i] for i in range(len(h_cuts)-1))
        max_v = max(v_cuts[i+1] - v_cuts[i] for i in range(len(v_cuts)-1))
        
        # The minimum cost will be the product of maximum distances
        # This is because each cut will need to be made max_h * max_v times
        # For horizontal cuts: each cut costs horizontalCut[i] and needs to be made max_v times
        # For vertical cuts: each cut costs verticalCut[j] and needs to be made max_h times
        
        total_cost = 0
        
        # Add cost of horizontal cuts
        for cost in horizontalCut:
            total_cost += cost * max_v
            
        # Add cost of vertical cuts
        for cost in verticalCut:
            total_cost += cost * max_h
            
        return total_cost