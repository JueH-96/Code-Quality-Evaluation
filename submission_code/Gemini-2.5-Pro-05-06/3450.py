class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # current_child_index stores the index of the child currently holding the ball.
        # Initially, child 0 holds the ball.
        current_child_index = 0
        
        # direction stores the direction of the next pass.
        # 1 means passing to the right (increasing index).
        # -1 means passing to the left (decreasing index).
        # Initially, the direction is to the right.
        direction = 1
        
        # We simulate k seconds. In each second, one pass occurs.
        for _ in range(k):
            # The child holding the ball passes it to the next child in the current direction.
            current_child_index += direction
            
            # Check if the ball has reached an end of the line.
            # If it reaches the rightmost child (n-1), the direction reverses.
            if current_child_index == n - 1:
                direction = -1
            # If it reaches the leftmost child (0), the direction reverses.
            # Note: using 'elif' because if n=2, child 0 and n-1 are distinct.
            # If n=1 (not allowed by constraints but good to consider), 0 and n-1 would be same.
            # Since n >= 2, 0 and n-1 are always distinct.
            elif current_child_index == 0:
                direction = 1
                
        # After k seconds (k passes), current_child_index is the child holding the ball.
        return current_child_index