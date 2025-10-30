class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = float('-inf')
        
        for start in range(n):
            current = start
            total = start
            
            # To handle large k, we will use a loop to find the cycle
            visited = {}
            sequence = []
            step = 0
            
            while step < k:
                if current in visited:
                    # Cycle detected
                    cycle_start = visited[current]
                    cycle_length = step - cycle_start
                    cycle_sum = sum(sequence[cycle_start:])
                    pre_cycle_sum = sum(sequence[:cycle_start])
                    
                    # Calculate how many full cycles we can take
                    remaining_steps = k - step
                    full_cycles = remaining_steps // cycle_length
                    total += pre_cycle_sum + full_cycles * cycle_sum
                    
                    # Add the remaining steps in the cycle
                    total += sum(sequence[cycle_start:cycle_start + remaining_steps % cycle_length])
                    break
                
                visited[current] = step
                sequence.append(current)
                total += receiver[current]
                current = receiver[current]
                step += 1
            
            max_value = max(max_value, total)
        
        return max_value