class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort both cost arrays in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Keep track of how many horizontal and vertical segments are formed
        # Initially, there is 1 horizontal piece (the whole cake) and 1 vertical piece (the whole cake)
        h_seg = 1
        v_seg = 1
        
        # Indices for horizontal and vertical cuts
        i = 0
        j = 0
        
        total_cost = 0
        
        # We have m - 1 horizontal cuts and n - 1 vertical cuts to account for
        while i < len(horizontalCut) or j < len(verticalCut):
            # If we still have both horizontal and vertical cuts to consider
            if i < len(horizontalCut) and j < len(verticalCut):
                # Pick the larger cost cut first (greedy choice)
                if horizontalCut[i] >= verticalCut[j]:
                    total_cost += horizontalCut[i] * v_seg
                    i += 1
                    h_seg += 1
                else:
                    total_cost += verticalCut[j] * h_seg
                    j += 1
                    v_seg += 1
            # If only horizontal cuts are left
            elif i < len(horizontalCut):
                total_cost += horizontalCut[i] * v_seg
                i += 1
                h_seg += 1
            # If only vertical cuts are left
            else:
                total_cost += verticalCut[j] * h_seg
                j += 1
                v_seg += 1
        
        return total_cost