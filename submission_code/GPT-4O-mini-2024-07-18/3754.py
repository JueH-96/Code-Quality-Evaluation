class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # Initialize the position at the origin
        x, y = 0, 0
        
        # Calculate the final position after all movements
        for char in s:
            if char == 'N':
                y += 1
            elif char == 'S':
                y -= 1
            elif char == 'E':
                x += 1
            elif char == 'W':
                x -= 1
        
        # Calculate the current Manhattan distance from the origin
        current_distance = abs(x) + abs(y)
        
        # The maximum distance we can achieve is the current distance plus the number of changes we can make
        # Each change can effectively add 1 to the distance in the best case
        max_distance = current_distance + k
        
        return max_distance