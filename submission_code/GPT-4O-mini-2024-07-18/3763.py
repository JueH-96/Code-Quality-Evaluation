class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        from bisect import bisect_left, bisect_right
        
        # Calculate the total area of all squares
        total_area = 0
        events = []
        
        for x, y, l in squares:
            total_area += l * l
            events.append((y, l, 1))  # Start of square
            events.append((y + l, l, -1))  # End of square
        
        # Sort events by y-coordinate
        events.sort()
        
        # To find the minimum y-coordinate
        current_area = 0
        last_y = 0
        
        # This will hold the area below the current y
        area_below = 0
        
        for i, (y, l, typ) in enumerate(events):
            # Calculate the area contribution from the last y to the current y
            if last_y < y:
                height = y - last_y
                area_below += current_area * height
            
            # Update the current area based on the type of event
            if typ == 1:  # Starting a square
                current_area += l * l
            else:  # Ending a square
                current_area -= l * l
            
            last_y = y
            
            # Check if we have found the balance point
            if area_below * 2 == total_area:
                return float(y)
            elif area_below * 2 < total_area:
                # We need to find the exact point where area_below == total_area / 2
                # We can use binary search or linear interpolation to find the exact point
                next_y = events[i + 1][0] if i + 1 < len(events) else y
                if next_y > y:
                    # Linear interpolation
                    required_area = total_area / 2
                    area_needed = required_area - area_below
                    if area_needed > 0:
                        # The height we need to find
                        height_needed = area_needed / current_area
                        return float(y + height_needed)
        
        return -1.0  # Should not reach here if input is valid