class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_sum = 0
        
        for start in range(n):
            # Detect cycle
            visited = {}
            path = []
            current = start
            
            while current not in visited:
                visited[current] = len(path)
                path.append(current)
                current = receiver[current]
            
            # current is now the first repeated node
            cycle_start_idx = visited[current]
            cycle_length = len(path) - cycle_start_idx
            
            # Calculate f(start) - we need to visit k+1 nodes
            total_nodes = k + 1
            
            if total_nodes <= cycle_start_idx:
                # We don't reach the cycle
                f_sum = sum(path[:total_nodes])
            else:
                # We reach the cycle
                # Sum of nodes before the cycle
                f_sum = sum(path[:cycle_start_idx])
                
                # Remaining nodes to visit after reaching the cycle
                remaining_nodes = total_nodes - cycle_start_idx
                
                # Sum of the cycle
                cycle_sum = sum(path[cycle_start_idx:])
                
                # Add complete cycles
                complete_cycles = remaining_nodes // cycle_length
                f_sum += complete_cycles * cycle_sum
                
                # Add remaining nodes in the cycle
                remaining = remaining_nodes % cycle_length
                for i in range(remaining):
                    f_sum += path[cycle_start_idx + i]
            
            max_sum = max(max_sum, f_sum)
        
        return max_sum