class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_steps = k.bit_length()
        
        # Pre-compute jumps and sums for each power of 2
        jump = [receiver]
        total = [receiver]
        for i in range(1, max_steps):
            next_jump = [jump[-1][jump[-1][j]] for j in range(n)]
            next_total = [total[-1][j] + total[-1][jump[-1][j]] for j in range(n)]
            jump.append(next_jump)
            total.append(next_total)
        
        max_value = 0
        for start in range(n):
            current = start
            value = start
            
            for i in range(max_steps):
                if k & (1 << i):
                    value += total[i][current]
                    current = jump[i][current]
            
            max_value = max(max_value, value)
        
        return max_value