class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        h_segments = 1
        v_segments = 1
        i = 0
        j = 0
        total_cost = 0
        
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] > verticalCut[j]:
                total_cost += horizontalCut[i] * v_segments
                h_segments += 1
                i += 1
            else:
                total_cost += verticalCut[j] * h_segments
                v_segments += 1
                j += 1
        
        # Add remaining horizontal cuts
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * v_segments
            h_segments += 1
            i += 1
        
        # Add remaining vertical cuts
        while j < len(verticalCut):
            total_cost += verticalCut[j] * h_segments
            v_segments += 1
            j += 1
        
        return total_cost