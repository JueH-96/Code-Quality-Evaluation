class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the number of 'R's and '_'s
        right_moves = moves.count('R') + moves.count('_')
        
        # The furthest point to the right is when all 'R's and '_'s are used to move right
        max_right = right_moves
        
        # The furthest point to the left is when all 'L's and '_'s are used to move left
        max_left = moves.count('L') + moves.count('_')
        
        # Return the maximum of the two distances
        return max(max_right, max_left)