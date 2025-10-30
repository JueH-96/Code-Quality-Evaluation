class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        # For each player as a starting point
        for start in range(n):
            current = start
            sum_ids = start
            passes = 0
            
            # Simulate k passes
            while passes < k:
                current = receiver[current]
                sum_ids += current
                passes += 1
            
            # Update the maximum value of f(x)
            max_value = max(max_value, sum_ids)
        
        return max_value