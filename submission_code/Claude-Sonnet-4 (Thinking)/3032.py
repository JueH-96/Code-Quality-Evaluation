class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        for start in range(n):
            # Simulate the path and detect cycle
            visited = {}
            path = []
            current = start
            step = 0
            
            while current not in visited:
                visited[current] = step
                path.append(current)
                current = receiver[current]
                step += 1
            
            # current is the start of the cycle
            cycle_start_index = visited[current]
            cycle_length = len(path) - cycle_start_index
            
            # Calculate f(start)
            total_sum = start
            remaining_k = k
            
            # Process the pre-cycle part
            pre_cycle_length = cycle_start_index
            if remaining_k <= pre_cycle_length:
                for i in range(remaining_k):
                    total_sum += receiver[path[i]]
            else:
                # Add all pre-cycle receivers
                for i in range(pre_cycle_length):
                    total_sum += receiver[path[i]]
                remaining_k -= pre_cycle_length
                
                # Process complete cycles
                cycle_sum = sum(receiver[path[i]] for i in range(cycle_start_index, len(path)))
                full_cycles = remaining_k // cycle_length
                total_sum += full_cycles * cycle_sum
                remaining_k %= cycle_length
                
                # Process remaining partial cycle
                for i in range(remaining_k):
                    total_sum += receiver[path[cycle_start_index + i]]
            
            max_value = max(max_value, total_sum)
        
        return max_value