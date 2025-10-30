class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort costs of horizontal and vertical cuts in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        total_cost = 0
        h_segments = 1  # Number of horizontal pieces so far
        v_segments = 1  # Number of vertical pieces so far
        
        # Indices for horizontal and vertical arrays
        i, j = 0, 0
        
        # Use a greedy approach by taking the largest cut cost first
        while i < len(horizontalCut) and j < len(verticalCut):
            # Compare the next largest horizontal vs. vertical cut
            if horizontalCut[i] >= verticalCut[j]:
                # If horizontal cut is larger or equal, use it
                # This cut splits all current vertical segments
                total_cost += horizontalCut[i] * v_segments
                h_segments += 1
                i += 1
            else:
                # If vertical cut is larger, use it
                # This cut splits all current horizontal segments
                total_cost += verticalCut[j] * h_segments
                v_segments += 1
                j += 1
        
        # Process remaining horizontal cuts (if any)
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * v_segments
            h_segments += 1
            i += 1
        
        # Process remaining vertical cuts (if any)
        while j < len(verticalCut):
            total_cost += verticalCut[j] * h_segments
            v_segments += 1
            j += 1
        
        return total_cost