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
            
            # Generate all possible next states
            next_states = []
            if current % 11 == 0:
                next_states.append(current // 11)
            if current % 5 == 0:
                next_states.append(current // 5)
            next_states.append(current - 1)
            next_states.append(current + 1)
            
            for next_x in next_states:
                if next_x == y:
                    return steps + 1
                if next_x >= 1 and next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, steps + 1))
        
        return -1  # This line is theoretically unreachable given problem constraints