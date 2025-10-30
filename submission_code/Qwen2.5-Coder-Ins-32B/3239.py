class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        queue = deque([(x, 0)])
        visited = set([x])
        
        while queue:
            current, operations = queue.popleft()
            
            if current == y:
                return operations
            
            # Decrement x by 1
            if current - 1 not in visited:
                visited.add(current - 1)
                queue.append((current - 1, operations + 1))
            
            # Increment x by 1
            if current + 1 not in visited:
                visited.add(current + 1)
                queue.append((current + 1, operations + 1))
            
            # Divide x by 11 if x is a multiple of 11
            if current % 11 == 0 and current // 11 not in visited:
                visited.add(current // 11)
                queue.append((current // 11, operations + 1))
            
            # Divide x by 5 if x is a multiple of 5
            if current % 5 == 0 and current // 5 not in visited:
                visited.add(current // 5)
                queue.append((current // 5, operations + 1))