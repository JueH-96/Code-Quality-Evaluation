class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        
        total_cost = 0
        h_cuts = len(horizontalCut) + 1
        v_cuts = len(verticalCut) + 1
        
        h_index = len(horizontalCut) - 1
        v_index = len(verticalCut) - 1
        
        # The number of pieces we have in the vertical and horizontal direction
        while h_index >= 0 and v_index >= 0:
            if horizontalCut[h_index] > verticalCut[v_index]:
                total_cost += horizontalCut[h_index] * v_cuts
                h_index -= 1
            else:
                total_cost += verticalCut[v_index] * h_cuts
                v_index -= 1
            
            # Update the number of pieces after a cut
            if h_index >= 0 and verticalCut[v_index] >= 0:
                v_cuts += 1
            if v_index >= 0 and horizontalCut[h_index] >= 0:
                h_cuts += 1
        
        # If there are remaining horizontal cuts
        while h_index >= 0:
            total_cost += horizontalCut[h_index] * v_cuts
            h_index -= 1
        
        # If there are remaining vertical cuts
        while v_index >= 0:
            total_cost += verticalCut[v_index] * h_cuts
            v_index -= 1
        
        return total_cost