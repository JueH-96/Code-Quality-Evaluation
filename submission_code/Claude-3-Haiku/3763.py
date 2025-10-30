class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Sort the squares by their y-coordinate
        squares.sort(key=lambda x: x[1])
        
        # Initialize the total area above and below the line
        total_area_above = 0
        total_area_below = 0
        
        # Iterate through the squares
        for x, y, l in squares:
            # Calculate the area of the current square
            area = l * l
            
            # Add the area to the appropriate total
            if y + l > total_area_above / l:
                total_area_above += area
            else:
                total_area_below += area
        
        # Find the y-coordinate where the areas are equal
        y = 0
        while abs(total_area_above - total_area_below) > 1e-5:
            if total_area_above > total_area_below:
                y += 1
                for x, y_coord, l in squares:
                    if y_coord + l > y:
                        total_area_above -= l * l
                    else:
                        total_area_below -= l * l
            else:
                y -= 1
                for x, y_coord, l in squares:
                    if y_coord + l > y:
                        total_area_above += l * l
                    else:
                        total_area_below += l * l
        
        return y