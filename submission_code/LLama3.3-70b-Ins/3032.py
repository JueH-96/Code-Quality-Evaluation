from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        for i in range(n):
            current_value = i
            current_player = i
            
            # Simulate the game for k passes
            for _ in range(k):
                current_player = receiver[current_player]
                current_value += current_player
            
            # Update the maximum value
            max_value = max(max_value, current_value)
        
        return max_value