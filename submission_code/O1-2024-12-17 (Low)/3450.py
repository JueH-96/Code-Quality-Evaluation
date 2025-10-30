class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        position = 0       # start with child 0
        direction = 1      # initial direction is to the right (towards n-1)
        
        for _ in range(k):
            # move ball one step in the current direction
            position += direction
            # if we hit the leftmost or rightmost child, reverse direction
            if position == 0 or position == n - 1:
                direction = -direction
        
        return position