from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        queue = deque()
        visited = set()
        queue.append((x, 0))
        visited.add(x)
        
        while queue:
            current, steps = queue.popleft()
            
            # Generate all possible next values
            next_vals = []
            # Increment
            next_vals.append(current + 1)
            # Decrement
            next_vals.append(current - 1)
            # Divide by 5 if possible
            if current % 5 == 0:
                next_vals.append(current // 5)
            # Divide by 11 if possible
            if current % 11 == 0:
                next_vals.append(current // 11)
            
            for val in next_vals:
                if val == y:
                    return steps + 1
                # Skip if negative to avoid unnecessary processing
                if val < 0:
                    continue
                if val not in visited:
                    visited.add(val)
                    queue.append((val, steps + 1))
        
        # The problem constraints ensure a solution exists, so this line is theoretically unreachable
        return -1