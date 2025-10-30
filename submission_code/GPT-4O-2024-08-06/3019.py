class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Initialize counters for left and right moves
        left_moves = 0
        right_moves = 0
        underscores = 0
        
        # Count the number of 'L', 'R', and '_' in the moves
        for move in moves:
            if move == 'L':
                left_moves += 1
            elif move == 'R':
                right_moves += 1
            elif move == '_':
                underscores += 1
        
        # Calculate the maximum possible distance
        # We can use underscores to either increase left or right moves
        max_distance = abs(left_moves - right_moves) + underscores
        
        return max_distance