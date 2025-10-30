class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # Initial position
        x, y = 0, 0
        
        # Track the net movements in both directions
        net_north_south = 0  # Positive for north, negative for south
        net_east_west = 0    # Positive for east, negative for west
        
        # Calculate the initial net movements
        for move in s:
            if move == 'N':
                net_north_south += 1
            elif move == 'S':
                net_north_south -= 1
            elif move == 'E':
                net_east_west += 1
            elif move == 'W':
                net_east_west -= 1
        
        # Maximum distance without any changes
        max_distance = abs(net_north_south) + abs(net_east_west)
        
        # If we have no changes allowed, return the max distance calculated
        if k == 0:
            return max_distance
        
        # We need to consider the best use of k changes to maximize the distance
        # We can either increase the most positive direction or decrease the most negative direction
        # Calculate the potential maximum distances by adjusting each direction
        potential_distances = []
        
        # Adjust north/south
        if net_north_south >= 0:
            # Increase north or decrease south
            potential_distances.append(net_north_south + k + abs(net_east_west))
        else:
            # Increase south or decrease north
            potential_distances.append(-net_north_south + k + abs(net_east_west))
        
        # Adjust east/west
        if net_east_west >= 0:
            # Increase east or decrease west
            potential_distances.append(net_east_west + k + abs(net_north_south))
        else:
            # Increase west or decrease east
            potential_distances.append(-net_east_west + k + abs(net_north_south))
        
        # The maximum possible distance we can achieve with up to k changes
        return max(potential_distances)