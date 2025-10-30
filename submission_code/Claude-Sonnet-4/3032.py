class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        for start in range(n):
            # Find cycle starting from 'start'
            visited = {}
            current = start
            path = []
            step = 0
            
            # Detect cycle
            while current not in visited:
                visited[current] = step
                path.append(current)
                current = receiver[current]
                step += 1
            
            # Now current is the start of the cycle
            cycle_start_idx = visited[current]
            pre_cycle = path[:cycle_start_idx]
            cycle = path[cycle_start_idx:]
            
            # Calculate f(start)
            total_sum = start
            remaining_k = k
            
            # Add pre-cycle part
            if remaining_k <= len(pre_cycle):
                # We don't even reach the cycle
                for i in range(remaining_k):
                    total_sum += receiver[path[i]]
            else:
                # Add all pre-cycle receivers
                for node in pre_cycle:
                    total_sum += receiver[node]
                remaining_k -= len(pre_cycle)
                
                # Now we're in the cycle
                cycle_sum = sum(cycle)
                full_cycles = remaining_k // len(cycle)
                remainder = remaining_k % len(cycle)
                
                # Add full cycles
                total_sum += full_cycles * cycle_sum
                
                # Add remainder
                for i in range(remainder):
                    total_sum += cycle[i]
            
            max_value = max(max_value, total_sum)
        
        return max_value