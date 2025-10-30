class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Combine all cuts with their type ('h' for horizontal, 'v' for vertical)
        cuts = []
        for cost in horizontalCut:
            cuts.append(('h', cost))
        for cost in verticalCut:
            cuts.append(('v', cost))
        
        # Sort the cuts in descending order of their cost
        cuts.sort(key=lambda x: -x[1])
        
        total_cost = 0
        h_count = 0  # Number of horizontal cuts made so far
        v_count = 0  # Number of vertical cuts made so far
        
        for cut_type, cost in cuts:
            if cut_type == 'h':
                total_cost += cost * (v_count + 1)
                h_count += 1
            else:
                total_cost += cost * (h_count + 1)
                v_count += 1
        
        return total_cost