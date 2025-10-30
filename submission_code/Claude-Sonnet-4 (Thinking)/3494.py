class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort cuts in descending order to prioritize expensive cuts
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        h = v = 0  # Indices for horizontal and vertical cuts made
        total_cost = 0
        
        # Greedily choose the more expensive cut at each step
        while h < len(horizontalCut) and v < len(verticalCut):
            if horizontalCut[h] >= verticalCut[v]:
                # Make horizontal cut across all existing vertical strips
                total_cost += horizontalCut[h] * (v + 1)
                h += 1
            else:
                # Make vertical cut across all existing horizontal strips
                total_cost += verticalCut[v] * (h + 1)
                v += 1
        
        # Make remaining horizontal cuts
        while h < len(horizontalCut):
            total_cost += horizontalCut[h] * (v + 1)
            h += 1
        
        # Make remaining vertical cuts
        while v < len(verticalCut):
            total_cost += verticalCut[v] * (h + 1)
            v += 1
        
        return total_cost