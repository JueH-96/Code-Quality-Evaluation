class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Start at position 0
        curr_pos = 0
        
        # For each command, update position
        for cmd in commands:
            if cmd == "RIGHT":
                curr_pos += 1
            elif cmd == "LEFT":
                curr_pos -= 1
            elif cmd == "DOWN":
                curr_pos += n
            else: # UP
                curr_pos -= n
                
        return curr_pos