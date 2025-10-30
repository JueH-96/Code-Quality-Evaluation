class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_sum = 0
        
        for start in range(n):
            # Find cycle using Floyd's algorithm
            visited = {}
            current = start
            position = 0
            path_sum = 0
            
            # Build path until we find a cycle
            while current not in visited and position <= k:
                visited[current] = position
                path_sum += current
                if position < k:
                    current = receiver[current]
                    position += 1
                else:
                    break
            
            if position > k:
                # No cycle encountered within k steps
                max_sum = max(max_sum, path_sum)
                continue
            
            # We found a cycle
            cycle_start = current
            cycle_start_pos = visited[cycle_start]
            
            # Calculate sum before cycle
            sum_before_cycle = 0
            current = start
            for i in range(cycle_start_pos):
                sum_before_cycle += current
                current = receiver[current]
            
            # Calculate cycle sum and length
            cycle_sum = 0
            cycle_length = 0
            cycle_node = cycle_start
            while True:
                cycle_sum += cycle_node
                cycle_node = receiver[cycle_node]
                cycle_length += 1
                if cycle_node == cycle_start:
                    break
            
            # Calculate total sum
            if k < cycle_start_pos:
                # k steps don't reach the cycle
                total_sum = 0
                current = start
                for i in range(k + 1):
                    total_sum += current
                    if i < k:
                        current = receiver[current]
            else:
                # k steps reach the cycle
                remaining_steps = k - cycle_start_pos
                full_cycles = remaining_steps // cycle_length
                partial_cycle_steps = remaining_steps % cycle_length
                
                # Sum for partial cycle
                partial_sum = 0
                current = cycle_start
                for i in range(partial_cycle_steps + 1):
                    partial_sum += current
                    if i < partial_cycle_steps:
                        current = receiver[current]
                
                total_sum = sum_before_cycle + full_cycles * cycle_sum + partial_sum
            
            max_sum = max(max_sum, total_sum)
        
        return max_sum