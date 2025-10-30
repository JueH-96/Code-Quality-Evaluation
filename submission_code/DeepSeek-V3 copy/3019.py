class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Initialize counters for 'L', 'R', and '_'
        l_count = 0
        r_count = 0
        underscore_count = 0
        
        # Iterate through the moves to count each type
        for move in moves:
            if move == 'L':
                l_count += 1
            elif move == 'R':
                r_count += 1
            elif move == '_':
                underscore_count += 1
        
        # Calculate the maximum distance
        # The distance is the absolute difference between the total left and right moves
        # We can choose to assign all underscores to either 'L' or 'R' to maximize the distance
        # So, the total left moves are l_count + underscore_count if we assign all underscores to 'L'
        # Or, the total right moves are r_count + underscore_count if we assign all underscores to 'R'
        # The maximum distance is the maximum of these two scenarios
        max_distance = max(abs(l_count + underscore_count - r_count), abs(r_count + underscore_count - l_count))
        
        return max_distance