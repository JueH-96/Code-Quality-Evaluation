from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        LOG = 34  # since k <= 1e10, 2^34 > 1e10
        # Initialize jump and sum tables
        jump = [receiver.copy()]
        summ = [receiver.copy()]
        
        for j in range(1, LOG):
            prev_jump = jump[j-1]
            new_jump = [prev_jump[prev_jump[x]] for x in range(n)]
            jump.append(new_jump)
            
            prev_sum = summ[j-1]
            new_sum = [prev_sum[x] + prev_sum[prev_jump[x]] for x in range(n)]
            summ.append(new_sum)
        
        max_f = 0
        for x in range(n):
            total = x
            current = x
            steps = k
            for j in range(LOG):
                if steps & 1:
                    total += summ[j][current]
                    current = jump[j][current]
                steps >>= 1
                if steps == 0:
                    break
            max_f = max(max_f, total)
        
        return max_f