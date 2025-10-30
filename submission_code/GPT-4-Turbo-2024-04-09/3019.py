class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        max_distance = 0
        current_position = 0
        
        for move in moves:
            if move == 'L':
                current_position -= 1
            elif move == 'R':
                current_position += 1
            else:  # move == '_'
                # We can choose to move either left or right
                # To maximize the distance, always move in the direction that increases the absolute value
                if current_position >= 0:
                    current_position += 1
                else:
                    current_position -= 1
            
            max_distance = max(max_distance, abs(current_position))
        
        return max_distance