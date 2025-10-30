class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        max_dist = 0
        n = len(s)
        
        # For each prefix, calculate the maximum possible Manhattan distance
        for i in range(n):
            prefix = s[:i+1]
            
            # Count of each direction in the prefix
            count_n = prefix.count('N')
            count_s = prefix.count('S')
            count_e = prefix.count('E')
            count_w = prefix.count('W')
            
            # Original position after following the prefix
            x = count_e - count_w
            y = count_n - count_s
            
            # Original Manhattan distance
            dist = abs(x) + abs(y)
            max_dist = max(max_dist, dist)
            
            # Maximum number of changes we can make for this prefix
            changes = min(k, i + 1)
            
            # Try all possible ways to allocate changes between the x and y dimensions
            for x_changes in range(changes + 1):
                y_changes = changes - x_changes
                
                # Compute the best possible x-coordinate after making 'x_changes' changes
                new_x = x
                if x > 0:
                    # If x > 0, change 'W' to 'E' to increase x further
                    new_x += min(x_changes, count_w) * 2
                elif x < 0:
                    # If x < 0, change 'E' to 'W' to decrease x further
                    new_x -= min(x_changes, count_e) * 2
                else:  # x == 0
                    # If x = 0, choose the direction that gives the maximum magnitude
                    w_to_e = min(x_changes, count_w)
                    e_to_w = min(x_changes, count_e)
                    if w_to_e >= e_to_w:
                        new_x = 2 * w_to_e
                    else:
                        new_x = -2 * e_to_w
                
                # Compute the best possible y-coordinate after making 'y_changes' changes
                new_y = y
                if y > 0:
                    # If y > 0, change 'S' to 'N' to increase y further
                    new_y += min(y_changes, count_s) * 2
                elif y < 0:
                    # If y < 0, change 'N' to 'S' to decrease y further
                    new_y -= min(y_changes, count_n) * 2
                else:  # y == 0
                    # If y = 0, choose the direction that gives the maximum magnitude
                    s_to_n = min(y_changes, count_s)
                    n_to_s = min(y_changes, count_n)
                    if s_to_n >= n_to_s:
                        new_y = 2 * s_to_n
                    else:
                        new_y = -2 * n_to_s
                
                # Update the maximum distance
                max_dist = max(max_dist, abs(new_x) + abs(new_y))
        
        return max_dist