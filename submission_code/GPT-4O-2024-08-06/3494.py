class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Initialize the total cost
        total_cost = 0
        
        # Initialize the number of pieces in each direction
        horizontal_pieces = 1
        vertical_pieces = 1
        
        # Use two pointers to iterate over the cuts
        h = 0
        v = 0
        
        # While there are cuts to be made
        while h < len(horizontalCut) and v < len(verticalCut):
            # Choose the more expensive cut
            if horizontalCut[h] >= verticalCut[v]:
                # Make a horizontal cut
                total_cost += horizontalCut[h] * vertical_pieces
                horizontal_pieces += 1
                h += 1
            else:
                # Make a vertical cut
                total_cost += verticalCut[v] * horizontal_pieces
                vertical_pieces += 1
                v += 1
        
        # If there are remaining horizontal cuts
        while h < len(horizontalCut):
            total_cost += horizontalCut[h] * vertical_pieces
            h += 1
        
        # If there are remaining vertical cuts
        while v < len(verticalCut):
            total_cost += verticalCut[v] * horizontal_pieces
            v += 1
        
        return total_cost