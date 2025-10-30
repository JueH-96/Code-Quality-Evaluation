class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Create a list of all cuts with their type
        cuts = []
        for cost in horizontalCut:
            cuts.append((cost, 'h'))
        for cost in verticalCut:
            cuts.append((cost, 'v'))
        
        # Sort cuts by cost in descending order (make expensive cuts first)
        cuts.sort(reverse=True)
        
        # Track number of horizontal and vertical pieces
        h_pieces = 1  # Initially 1 piece horizontally
        v_pieces = 1  # Initially 1 piece vertically
        
        total_cost = 0
        
        for cost, cut_type in cuts:
            if cut_type == 'h':
                # Horizontal cut goes through all vertical pieces
                total_cost += cost * v_pieces
                h_pieces += 1
            else:  # cut_type == 'v'
                # Vertical cut goes through all horizontal pieces
                total_cost += cost * h_pieces
                v_pieces += 1
        
        return total_cost