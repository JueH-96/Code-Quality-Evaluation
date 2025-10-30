class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Helper function to check if valid cuts can be made in one dimension (either x or y)
        def canMakeCuts(coords, n):
            # Sort the intervals based on start coordinate
            coords.sort()
            # We need to find two cuts that divide the rectangles into three non-empty groups
            # We use a greedy approach to find the first cut
            first_cut = -1
            count = 0
            for i, (start, end) in enumerate(coords):
                if first_cut == -1 or start <= first_cut:
                    count += 1
                    first_cut = max(first_cut, end)
                else:
                    break
            if count == 0 or count == len(coords):
                return False
            
            # After the first cut, find the second cut
            second_cut = -1
            for j in range(i, len(coords)):
                start, end = coords[j]
                if second_cut == -1 or start <= second_cut:
                    second_cut = max(second_cut, end)
                else:
                    break
            
            # Check if all rectangles are covered and are in three distinct parts
            return j < len(coords) and second_cut < n
        
        # Extract x and y coordinates intervals
        x_coords = [(rect[0], rect[2]) for rect in rectangles]
        y_coords = [(rect[1], rect[3]) for rect in rectangles]
        
        # Check for both vertical and horizontal cuts
        return canMakeCuts(x_coords, n) or canMakeCuts(y_coords, n)

# The code defines a class Solution with a method checkValidCuts.
# It uses a helper function canMakeCuts to determine if valid cuts can be made in one dimension.
# It checks both dimensions (x and y) to see if the required cuts can be made.