class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        if x == y:
            return 0
        
        visited = set()
        queue = deque([(x, 0)])  # (current value of x, number of operations)
        visited.add(x)
        
        while queue:
            current, operations = queue.popleft()
            
            # Generate possible next states
            next_states = []
            if current > 1:
                next_states.append(current - 1)  # Decrement
            next_states.append(current + 1)      # Increment
            
            if current % 5 == 0:
                next_states.append(current // 5)  # Divide by 5
            if current % 11 == 0:
                next_states.append(current // 11)  # Divide by 11
            
            for next_x in next_states:
                if next_x == y:
                    return operations + 1
                if 1 <= next_x <= 10000 and next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))
        
        return -1  # This line should never be reached due to problem constraints