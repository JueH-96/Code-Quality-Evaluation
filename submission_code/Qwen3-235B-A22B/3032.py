class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Determine the maximum level needed for binary lifting
        max_level = 0
        while (1 << max_level) <= k:
            max_level += 1
        max_level -= 1
        
        # Initialize up and sum tables
        up = [[0] * n for _ in range(max_level + 1)]
        sum_ = [[0] * n for _ in range(max_level + 1)]
        
        # Fill the base level (j=0)
        for u in range(n):
            up[0][u] = receiver[u]
            sum_[0][u] = receiver[u]
        
        # Fill higher levels using dynamic programming
        for j in range(1, max_level + 1):
            for u in range(n):
                v = up[j-1][u]
                up[j][u] = up[j-1][v]
                sum_[j][u] = sum_[j-1][u] + sum_[j-1][v]
        
        # Calculate the maximum function value
        max_value = 0
        for x in range(n):
            current_node = x
            total_sum = 0
            remaining = k
            # Process each bit from highest to lowest
            for j in reversed(range(max_level + 1)):
                if remaining >= (1 << j):
                    total_sum += sum_[j][current_node]
                    remaining -= (1 << j)
                    current_node = up[j][current_node]
            # Update maximum value
            current_value = x + total_sum
            if current_value > max_value:
                max_value = current_value
        return max_value