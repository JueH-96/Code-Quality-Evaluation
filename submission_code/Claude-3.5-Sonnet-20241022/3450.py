class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Initialize position and direction
        pos = 0
        right = True
        
        # For each second
        for _ in range(k):
            # If going right
            if right:
                # If at rightmost position, change direction and move left
                if pos == n-1:
                    right = False
                    pos -= 1
                # Otherwise keep moving right
                else:
                    pos += 1
            # If going left
            else:
                # If at leftmost position, change direction and move right
                if pos == 0:
                    right = True
                    pos += 1
                # Otherwise keep moving left
                else:
                    pos -= 1
                    
        return pos