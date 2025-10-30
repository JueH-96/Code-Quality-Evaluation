class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        h_cuts = 1
        v_cuts = 1
        
        total_cost = 0
        
        h_idx = 0
        v_idx = 0
        
        while h_idx < len(horizontalCut) and v_idx < len(verticalCut):
            if horizontalCut[h_idx] > verticalCut[v_idx]:
                total_cost += horizontalCut[h_idx] * v_cuts
                h_cuts += 1
                h_idx += 1
            else:
                total_cost += verticalCut[v_idx] * h_cuts
                v_cuts += 1
                v_idx += 1
        
        while h_idx < len(horizontalCut):
            total_cost += horizontalCut[h_idx] * v_cuts
            h_cuts += 1
            h_idx += 1
            
        while v_idx < len(verticalCut):
            total_cost += verticalCut[v_idx] * h_cuts
            v_cuts += 1
            v_idx += 1
            
        return total_cost