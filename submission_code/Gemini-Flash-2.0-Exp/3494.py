class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        h_segments = 1
        v_segments = 1
        
        h_idx = 0
        v_idx = 0
        
        total_cost = 0
        
        while h_idx < len(horizontalCut) or v_idx < len(verticalCut):
            if h_idx < len(horizontalCut) and (v_idx == len(verticalCut) or horizontalCut[h_idx] > verticalCut[v_idx]):
                total_cost += horizontalCut[h_idx] * v_segments
                h_segments += 1
                h_idx += 1
            else:
                total_cost += verticalCut[v_idx] * h_segments
                v_segments += 1
                v_idx += 1
        
        return total_cost