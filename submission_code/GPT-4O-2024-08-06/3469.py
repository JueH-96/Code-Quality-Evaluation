class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Initialize the height of the triangle
        height = 0
        
        # While we can still form a new row
        while True:
            # Calculate the number of balls needed for the next row
            needed_balls = height + 1
            
            # Check if we can use red balls for this row
            if height % 2 == 0:
                if red >= needed_balls:
                    red -= needed_balls
                else:
                    break
            # Check if we can use blue balls for this row
            else:
                if blue >= needed_balls:
                    blue -= needed_balls
                else:
                    break
            
            # Increment the height as we have successfully formed this row
            height += 1
        
        return height