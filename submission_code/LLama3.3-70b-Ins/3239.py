from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        queue = deque([(x, 0)])
        visited = set([x])
        
        while queue:
            current, steps = queue.popleft()
            
            if current == y:
                return steps
            
            # Divide by 11
            if current % 11 == 0 and current // 11 not in visited:
                queue.append((current // 11, steps + 1))
                visited.add(current // 11)
            
            # Divide by 5
            if current % 5 == 0 and current // 5 not in visited:
                queue.append((current // 5, steps + 1))
                visited.add(current // 5)
            
            # Decrement by 1
            if current - 1 > 0 and current - 1 not in visited:
                queue.append((current - 1, steps + 1))
                visited.add(current - 1)
            
            # Increment by 1
            if current + 1 <= 10**4 and current + 1 not in visited:
                queue.append((current + 1, steps + 1))
                visited.add(current + 1)
        
        return -1