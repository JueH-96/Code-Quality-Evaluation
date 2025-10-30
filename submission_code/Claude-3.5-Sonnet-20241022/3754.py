class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        # Function to calculate Manhattan distance from origin
        def manhattan_dist(x, y):
            return abs(x) + abs(y)
        
        # Function to get the opposite direction
        def get_opposite(dir):
            if dir == 'N': return 'S'
            if dir == 'S': return 'N'
            if dir == 'E': return 'W'
            return 'E'
        
        # Function to calculate position after a move
        def get_pos_change(dir):
            if dir == 'N': return (0, 1)
            if dir == 'S': return (0, -1)
            if dir == 'E': return (1, 0)
            return (-1, 0)
        
        # Calculate original path positions
        positions = [(0, 0)]
        curr_x, curr_y = 0, 0
        for char in s:
            dx, dy = get_pos_change(char)
            curr_x += dx
            curr_y += dy
            positions.append((curr_x, curr_y))
        
        max_dist = 0
        # For each position, try to maximize distance by changing k characters
        for i in range(n + 1):
            x, y = positions[i]
            # Calculate contribution of unchanged characters before position i
            contrib_before = 0
            curr_x, curr_y = 0, 0
            for j in range(i):
                dx, dy = get_pos_change(s[j])
                curr_x += dx
                curr_y += dy
            
            # Try changing up to k characters after position i
            remaining_moves = n - i
            changes = min(k, remaining_moves)
            
            # For remaining positions after i, choose the direction that maximizes distance
            # We can change up to k characters to their opposite to maximize distance
            x_after, y_after = curr_x, curr_y
            
            # Count original movements after position i
            for j in range(i, n):
                dx, dy = get_pos_change(s[j])
                x_after += dx
                y_after += dy
            
            # For each position after i, we can change up to k characters
            # to maximize the distance from origin
            if x_after >= 0:
                x_after += changes
            else:
                x_after -= changes
                
            if y_after >= 0:
                y_after += changes
            else:
                y_after -= changes
                
            max_dist = max(max_dist, manhattan_dist(x_after, y_after))
            
        return max_dist