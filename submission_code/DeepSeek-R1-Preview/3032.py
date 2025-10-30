class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_level = 40  # Since 2^40 > 1e10, which is the max k
        
        # Precompute jump and sum for each level
        jump = [[0] * n for _ in range(max_level)]
        sum_step = [[0] * n for _ in range(max_level)]
        
        # Initialize for level 0 (2^0 = 1 step)
        for x in range(n):
            jump[0][x] = receiver[x]
            sum_step[0][x] = receiver[x]
        
        # Fill the jump and sum arrays for higher levels
        for m in range(1, max_level):
            for x in range(n):
                prev_jump = jump[m-1][x]
                jump[m][x] = jump[m-1][prev_jump]
                sum_step[m][x] = sum_step[m-1][x] + sum_step[m-1][prev_jump]
        
        max_f = 0
        for x in range(n):
            current_x = x
            current_sum = 0
            remaining = k
            # Traverse from highest bit to lowest
            for m in reversed(range(max_level)):
                if remaining >= (1 << m):
                    current_sum += sum_step[m][current_x]
                    current_x = jump[m][current_x]
                    remaining -= (1 << m)
            # f(x) is x plus the sum of the k steps
            f_x = x + current_sum
            if f_x > max_f:
                max_f = f_x
        return max_f