class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        from collections import deque
        
        # Set upper bound to handle cases where we might need to go above max(x,y)
        upper_bound = 11 * max(x, y)
        
        queue = deque([(x, 0)])  # (current_value, operations_count)
        visited = set([x])
        
        while queue:
            current, ops = queue.popleft()
            
            # Generate all possible next states
            next_states = []
            
            # Divide by 11 if possible
            if current % 11 == 0:
                next_states.append(current // 11)
            
            # Divide by 5 if possible
            if current % 5 == 0:
                next_states.append(current // 5)
            
            # Decrement by 1 (don't go below 1)
            if current > 1:
                next_states.append(current - 1)
            
            # Increment by 1 (with upper bound)
            if current < upper_bound:
                next_states.append(current + 1)
            
            # Process each next state
            for next_val in next_states:
                if next_val == y:
                    return ops + 1
                
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))
        
        return -1  # Should never reach here given the constraints