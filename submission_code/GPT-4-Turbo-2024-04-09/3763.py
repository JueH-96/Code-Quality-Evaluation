class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        import bisect
        
        # Collect all the y-coordinates where events happen (start or end of a square)
        events = []
        for x, y, l in squares:
            events.append((y, l, 1))  # Starting event
            events.append((y + l, l, -1))  # Ending event
        
        # Sort events primarily by y-coordinate
        events.sort()
        
        # Total area of all squares
        total_area = sum(l * l for _, _, l in squares)
        
        # We need to find the y-coordinate where half of the area is below it
        half_area = total_area / 2
        
        # Sweep line algorithm to calculate the area below the current y-coordinate
        current_area = 0
        last_y = 0
        active_length = 0
        
        for i, event in enumerate(events):
            y, length, typ = event
            
            # Calculate the area contribution from the last_y to the current y
            if y != last_y:
                current_area += active_length * (y - last_y)
                last_y = y
            
            # Check if we've reached or passed half the area
            if current_area >= half_area:
                # We need to find the exact y-coordinate where the area is half
                # Since we might have jumped over it, calculate the exact position
                excess_area = current_area - half_area
                return y - excess_area / active_length
            
            # Update the active length of the squares
            active_length += length * typ
        
        # If we exit the loop without finding the exact y, it means the balance is at the top
        return events[-1][0]