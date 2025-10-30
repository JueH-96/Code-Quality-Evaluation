class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Initialize counters for left and right moves
        left_moves = right_moves = 0
        
        # Iterate through the moves string
        for move in moves:
            # Increment left_moves for 'L' or '_'
            if move == 'L' or move == '_':
                left_moves += 1
            # Increment right_moves for 'R' or '_'
            if move == 'R' or move == '_':
                right_moves += 1
        
        # The furthest distance is the maximum of left_moves and right_moves
        return max(left_moves, right_moves)