class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # Initial position
        x, y = 0, 0
        
        # Calculate initial position after all moves
        for char in s:
            if char == 'N':
                y += 1
            elif char == 'S':
                y -= 1
            elif char == 'E':
                x += 1
            elif char == 'W':
                x -= 1
        
        # Calculate the initial Manhattan distance
        initial_distance = abs(x) + abs(y)
        
        # Calculate the number of moves in each direction
        north_count = s.count('N')
        south_count = s.count('S')
        east_count = s.count('E')
        west_count = s.count('W')
        
        # Calculate the net north-south and east-west movements
        net_ns = north_count - south_count
        net_ew = east_count - west_count
        
        # Calculate the maximum possible distance after changing k moves
        max_distance = initial_distance + 2 * k
        
        # If we have more changes than needed to reverse all net movements, adjust
        if k >= abs(net_ns) + abs(net_ew):
            # We can reverse all net movements and use remaining changes to move further
            max_distance = abs(net_ns) + abs(net_ew) + 2 * (k - abs(net_ns) - abs(net_ew))
        
        return max_distance