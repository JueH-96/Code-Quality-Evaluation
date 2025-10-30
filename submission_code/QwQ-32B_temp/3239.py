from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        visited = set()
        queue = deque()
        queue.append((x, 0))
        visited.add(x)
        
        while queue:
            current, steps = queue.popleft()
            
            if current == y:
                return steps
            
            # Generate next states
            # Division by 11
            if current % 11 == 0:
                next_val = current // 11
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            
            # Division by 5
            if current % 5 == 0:
                next_val = current // 5
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            
            # Increment
            next_val = current + 1
            if next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
            
            # Decrement
            next_val = current - 1
            if next_val >= 1 and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
        
        return -1  # This line is theoretically unreachable as per problem constraints