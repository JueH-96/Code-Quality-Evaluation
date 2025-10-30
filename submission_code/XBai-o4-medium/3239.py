from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        visited = set()
        queue = deque()
        queue.append((x, 0))
        visited.add(x)
        
        while queue:
            current, steps = queue.popleft()
            
            next_vals = []
            # Check division by 11
            if current % 11 == 0 and current != 0:
                next_vals.append(current // 11)
            # Check division by 5
            if current % 5 == 0 and current != 0:
                next_vals.append(current // 5)
            # Decrement and increment
            next_vals.append(current - 1)
            next_vals.append(current + 1)
            
            for next_val in next_vals:
                if next_val == y:
                    return steps + 1
                # Check validity and add to queue if not visited
                if next_val >= 0 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
        
        return -1  # The problem constraints ensure this is unreachable