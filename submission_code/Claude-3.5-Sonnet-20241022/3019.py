class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left = right = underscore = 0
        
        # Count occurrences of each character
        for c in moves:
            if c == 'L':
                left += 1
            elif c == 'R':
                right += 1
            else:
                underscore += 1
                
        # Get the difference between left and right moves
        net_direction = abs(left - right)
        
        # All underscores should go in the direction that takes us furthest
        return net_direction + underscore