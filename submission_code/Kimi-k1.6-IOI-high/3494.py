class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Combine all cuts with their direction (0 for horizontal, 1 for vertical)
        cuts = []
        for cost in horizontalCut:
            cuts.append((cost, 0))
        for cost in verticalCut:
            cuts.append((cost, 1))
        
        # Sort cuts by their cost in descending order
        cuts.sort(reverse=True, key=lambda x: x[0])
        
        total_cost = 0
        h_count = 0  # Number of horizontal cuts made so far
        v_count = 0  # Number of vertical cuts made so far
        
        for cost, direction in cuts:
            if direction == 0:
                # Horizontal cut: multiply by (v_count + 1)
                total_cost += (v_count + 1) * cost
                h_count += 1
            else:
                # Vertical cut: multiply by (h_count + 1)
                total_cost += (h_count + 1) * cost
                v_count += 1
        
        return total_cost