class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        if x == y:
            return 0
        
        queue = deque([(x, 0)])
        visited = set([x])
        
        while queue:
            current, steps = queue.popleft()
            
            # Generate all possible next steps
            candidates = []
            if current % 11 == 0:
                candidates.append(current // 11)
            if current % 5 == 0:
                candidates.append(current // 5)
            candidates.append(current - 1)
            candidates.append(current + 1)
            
            for next_val in candidates:
                if next_val == y:
                    return steps + 1
                if next_val > 0 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
        
        return -1  # Should never reach here if constraints are correct