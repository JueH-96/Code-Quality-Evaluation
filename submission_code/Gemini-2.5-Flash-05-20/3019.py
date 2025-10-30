class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_fixed = 0
        right_fixed = 0
        wildcard_count = 0

        # Count the occurrences of 'L', 'R', and '_'
        for move in moves:
            if move == 'L':
                left_fixed += 1
            elif move == 'R':
                right_fixed += 1
            else: # move == '_'
                wildcard_count += 1
        
        # Calculate the net displacement from the fixed 'L' and 'R' moves
        fixed_net_displacement = right_fixed - left_fixed
        
        # Scenario 1: Assume all '_' moves are chosen to go right.
        # This will give the maximum possible final position value.
        max_possible_position = fixed_net_displacement + wildcard_count
        
        # Scenario 2: Assume all '_' moves are chosen to go left.
        # This will give the minimum possible final position value.
        min_possible_position = fixed_net_displacement - wildcard_count
        
        # The furthest distance from the origin is the maximum of the
        # absolute values of these two extreme reachable positions.
        return max(abs(max_possible_position), abs(min_possible_position))