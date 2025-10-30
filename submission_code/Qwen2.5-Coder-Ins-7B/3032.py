class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        for start in range(n):
            current_value = start
            current_player = start
            for _ in range(k):
                current_player = receiver[current_player]
                current_value += current_player
            max_value = max(max_value, current_value)
        
        return max_value