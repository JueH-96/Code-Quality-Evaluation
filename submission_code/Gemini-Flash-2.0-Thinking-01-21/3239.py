from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        q = deque([(x, 0)])
        visited = {x}
        while q:
            current_x, operations = q.popleft()
            if current_x == y:
                return operations
            
            # Divide by 11
            if current_x % 11 == 0:
                next_x = current_x // 11
                if next_x >= 0 and next_x not in visited:
                    visited.add(next_x)
                    q.append((next_x, operations + 1))
                    
            # Divide by 5
            if current_x % 5 == 0:
                next_x = current_x // 5
                if next_x >= 0 and next_x not in visited:
                    visited.add(next_x)
                    q.append((next_x, operations + 1))
                    
            # Decrement by 1
            next_x = current_x - 1
            if next_x >= 0 and next_x not in visited:
                visited.add(next_x)
                q.append((next_x, operations + 1))
                
            # Increment by 1
            next_x = current_x + 1
            if next_x <= 20000 and next_x not in visited: # Limit upper bound to avoid infinite search, 20000 is a safe upper bound based on problem constraints.
                visited.add(next_x)
                q.append((next_x, operations + 1))
        return -1 # Should not reach here as a solution is always possible within the constraints.