class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Combine all cuts with their types
        cuts = [(cost, 'h') for cost in horizontalCut] + [(cost, 'v') for cost in verticalCut]
        
        # Sort by cost in descending order (highest cost first)
        cuts.sort(reverse=True)
        
        # Track number of pieces in each direction
        h_pieces = 1  # horizontal pieces (rows)
        v_pieces = 1  # vertical pieces (columns)
        total_cost = 0
        
        for cost, cut_type in cuts:
            if cut_type == 'h':
                # Horizontal cut goes through all vertical pieces
                total_cost += cost * v_pieces
                h_pieces += 1
            else:
                # Vertical cut goes through all horizontal pieces
                total_cost += cost * h_pieces
                v_pieces += 1
        
        return total_cost