class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        
        # Precompute the next positions and sums for powers of 2 up to the largest power of 2 <= k
        max_power = k.bit_length()
        next_pos = [[0] * n for _ in range(max_power)]
        sum_pos = [[0] * n for _ in range(max_power)]
        
        for i in range(n):
            next_pos[0][i] = receiver[i]
            sum_pos[0][i] = receiver[i]
        
        for p in range(1, max_power):
            for i in range(n):
                next_pos[p][i] = next_pos[p-1][next_pos[p-1][i]]
                sum_pos[p][i] = sum_pos[p-1][i] + sum_pos[p-1][next_pos[p-1][i]]
        
        def compute_f(x):
            total_sum = x
            current_pos = x
            remaining_k = k
            
            for p in range(max_power):
                if remaining_k & (1 << p):
                    total_sum += sum_pos[p][current_pos]
                    current_pos = next_pos[p][current_pos]
            
            return total_sum
        
        max_value = 0
        for i in range(n):
            max_value = max(max_value, compute_f(i))
        
        return max_value