class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        if x <= y:
            return y - x
        
        queue = deque([(x, 0)])  # (current_value, operations_count)
        visited = {x}
        
        while queue:
            current, ops = queue.popleft()
            
            if current == y:
                return ops
            
            # Generate next possible states
            next_states = []
            
            # Divide by 11 if possible
            if current % 11 == 0:
                next_states.append(current // 11)
            
            # Divide by 5 if possible
            if current % 5 == 0:
                next_states.append(current // 5)
            
            # Decrement by 1
            if current > 1:  # Don't go below 1 as per constraints
                next_states.append(current - 1)
            
            # Increment by 1
            # We need an upper bound to avoid infinite exploration
            # The maximum useful value would be around 2*y or x+y
            if current < 2 * y + 50:  # Adding some buffer
                next_states.append(current + 1)
            
            for next_val in next_states:
                if next_val not in visited and next_val >= 1:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))
        
        return -1  # Should never reach here given the constraints