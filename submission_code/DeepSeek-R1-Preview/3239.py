from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        visited = set()
        queue = deque([(x, 0)])
        visited.add(x)
        
        while queue:
            current, steps = queue.popleft()
            
            if current == y:
                return steps
            
            # Generate all possible next steps
            next_x = current // 11
            if current % 11 == 0 and next_x not in visited:
                visited.add(next_x)
                queue.append((next_x, steps + 1))
            
            next_x = current // 5
            if current % 5 == 0 and next_x not in visited:
                visited.add(next_x)
                queue.append((next_x, steps + 1))
            
            next_x = current - 1
            if next_x >= 0 and next_x not in visited:
                visited.add(next_x)
                queue.append((next_x, steps + 1))
            
            next_x = current + 1
            if next_x not in visited:
                visited.add(next_x)
                queue.append((next_x, steps + 1))
        
        return -1  # This line is theoretically unreachable due to problem constraints