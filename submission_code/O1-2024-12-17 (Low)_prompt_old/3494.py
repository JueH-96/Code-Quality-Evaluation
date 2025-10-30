class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort both cost arrays in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        total_cost = 0
        
        # Number of horizontal pieces and vertical pieces we currently have
        # Initially, there's 1 horizontal "strip" and 1 vertical "strip"
        horizontal_slices = 1
        vertical_slices = 1
        
        # Pointers to traverse the cut cost arrays
        i, j = 0, 0
        
        # While there are cuts left in both arrays, pick the largest cost
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] >= verticalCut[j]:
                # Make a horizontal cut; cost is horizontalCut[i] times the current number of vertical slices
                total_cost += horizontalCut[i] * vertical_slices
                horizontal_slices += 1  # Increase the number of horizontal pieces
                i += 1
            else:
                # Make a vertical cut; cost is verticalCut[j] times the current number of horizontal slices
                total_cost += verticalCut[j] * horizontal_slices
                vertical_slices += 1  # Increase the number of vertical pieces
                j += 1
        
        # Add remaining horizontal cuts if any
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * vertical_slices
            i += 1
            horizontal_slices += 1
        
        # Add remaining vertical cuts if any
        while j < len(verticalCut):
            total_cost += verticalCut[j] * horizontal_slices
            j += 1
            vertical_slices += 1
        
        return total_cost