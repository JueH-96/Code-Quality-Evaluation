from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Combine the cuts into a list of tuples with their type
        cuts = []
        for h in horizontalCut:
            cuts.append((h, 'h'))
        for v in verticalCut:
            cuts.append((v, 'v'))
        
        # Sort the cuts in descending order of their cost
        cuts.sort(key=lambda x: -x[0])
        
        total = 0
        h_count = 0
        v_count = 0
        
        for cost, typ in cuts:
            if typ == 'h':
                # Horizontal cut: multiply by (number of vertical cuts + 1)
                total += cost * (v_count + 1)
                h_count += 1
            else:
                # Vertical cut: multiply by (number of horizontal cuts + 1)
                total += cost * (h_count + 1)
                v_count += 1
        
        return total