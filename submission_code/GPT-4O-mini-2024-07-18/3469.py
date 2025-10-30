class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Ensure red is the larger or equal count
        if red < blue:
            red, blue = blue, red
        
        height = 0
        while True:
            # Calculate the number of balls needed for the next row
            needed = height + 1
            if needed > red:
                break
            # Alternate colors for the next row
            height += 1
            red -= needed
            
            # Check if we can use blue for the next row
            needed = height + 1
            if needed > blue:
                break
            height += 1
            blue -= needed
        
        return height