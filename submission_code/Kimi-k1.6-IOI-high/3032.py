class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_level = 60  # Sufficient for k up to 1e18
        
        # Initialize binary lifting tables
        next_node = [[-1] * (max_level + 1) for _ in range(n)]
        sum_steps = [[0] * (max_level + 1) for _ in range(n)]
        
        # Base case: 2^0 steps
        for x in range(n):
            next_node[x][0] = receiver[x]
            sum_steps[x][0] = receiver[x]
        
        # Fill the binary lifting tables for higher powers of two
        for j in range(1, max_level + 1):
            for x in range(n):
                next_node[x][j] = next_node[next_node[x][j-1]][j-1]
                sum_steps[x][j] = sum_steps[x][j-1] + sum_steps[next_node[x][j-1]][j-1]
        
        max_sum = 0
        for x in range(n):
            current_node = x
            total_sum = 0
            remaining = k
            # Decompose k into powers of two and accumulate the sum
            for j in range(max_level, -1, -1):
                if remaining >= (1 << j):
                    total_sum += sum_steps[current_node][j]
                    current_node = next_node[current_node][j]
                    remaining -= (1 << j)
            # Add the starting node x to the accumulated sum
            current_sum = x + total_sum
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum