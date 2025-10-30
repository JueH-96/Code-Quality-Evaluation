class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order to prioritize the most expensive cuts first
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Initialize the number of horizontal and vertical pieces
        h_pieces = 1
        v_pieces = 1
        
        total_cost = 0
        
        # Merge the two lists and process them in descending order
        i = j = 0
        while i < len(horizontalCut) or j < len(verticalCut):
            if i < len(horizontalCut) and (j >= len(verticalCut) or horizontalCut[i] >= verticalCut[j]):
                total_cost += horizontalCut[i] * v_pieces
                h_pieces += 1
                i += 1
            else:
                total_cost += verticalCut[j] * h_pieces
                v_pieces += 1
                j += 1
        
        return total_cost