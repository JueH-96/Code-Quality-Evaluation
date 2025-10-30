class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        for x in range(n):
            curr_sum = x
            curr_player = x
            
            for _ in range(k):
                curr_player = receiver[curr_player]
                curr_sum += curr_player
            
            max_value = max(max_value, curr_sum)
        
        return max_value