class Solution:
    def getMaxFunctionValue(self, receiver: list[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        # Precompute the sum of the first k receivers for each starting player
        for start_player in range(n):
            current_player = start_player
            current_sum = start_player
            for _ in range(k):
                current_player = receiver[current_player]
                current_sum += current_player
            max_value = max(max_value, current_sum)
        
        return max_value