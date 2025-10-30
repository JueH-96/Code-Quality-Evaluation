class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort cuts in descending order (most expensive first)
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        total_cost = 0
        h_pieces = 1  # number of horizontal pieces
        v_pieces = 1  # number of vertical pieces
        h_idx = 0     # index for horizontal cuts
        v_idx = 0     # index for vertical cuts
        
        # Process all cuts, always choosing the more expensive one
        while h_idx < len(horizontalCut) and v_idx < len(verticalCut):
            if horizontalCut[h_idx] >= verticalCut[v_idx]:
                # Make horizontal cut
                # This cut will be applied to all vertical pieces
                total_cost += horizontalCut[h_idx] * v_pieces
                h_pieces += 1
                h_idx += 1
            else:
                # Make vertical cut
                # This cut will be applied to all horizontal pieces
                total_cost += verticalCut[v_idx] * h_pieces
                v_pieces += 1
                v_idx += 1
        
        # Process remaining horizontal cuts
        while h_idx < len(horizontalCut):
            total_cost += horizontalCut[h_idx] * v_pieces
            h_pieces += 1
            h_idx += 1
        
        # Process remaining vertical cuts
        while v_idx < len(verticalCut):
            total_cost += verticalCut[v_idx] * h_pieces
            v_pieces += 1
            v_idx += 1
        
        return total_cost