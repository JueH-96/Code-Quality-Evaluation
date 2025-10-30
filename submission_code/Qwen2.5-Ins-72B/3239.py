from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        visited = set()
        queue = deque([(x, 0)])
        
        while queue:
            current, operations = queue.popleft()
            
            if current == y:
                return operations
            
            if current in visited:
                continue
            visited.add(current)
            
            if current % 11 == 0:
                queue.append((current // 11, operations + 1))
            if current % 5 == 0:
                queue.append((current // 5, operations + 1))
            queue.append((current - 1, operations + 1))
            queue.append((current + 1, operations + 1))
        
        return -1  # This line should never be reached if the input is valid