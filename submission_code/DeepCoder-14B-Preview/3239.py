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
            
            # Generate next states by adding 1
            next_add = current + 1
            if next_add not in visited:
                if next_add == y:
                    return steps + 1
                visited.add(next_add)
                queue.append((next_add, steps + 1))
            
            # Generate next states by subtracting 1
            next_sub = current - 1
            if next_sub >= 1:  # Ensure we don't go below 1
                if next_sub not in visited:
                    if next_sub == y:
                        return steps + 1
                    visited.add(next_sub)
                    queue.append((next_sub, steps + 1))
            
            # Generate next states by dividing by 11 if possible
            if current % 11 == 0:
                next_div11 = current // 11
                if next_div11 not in visited:
                    if next_div11 == y:
                        return steps + 1
                    visited.add(next_div11)
                    queue.append((next_div11, steps + 1))
            
            # Generate next states by dividing by 5 if possible
            if current % 5 == 0:
                next_div5 = current // 5
                if next_div5 not in visited:
                    if next_div5 == y:
                        return steps + 1
                    visited.add(next_div5)
                    queue.append((next_div5, steps + 1))
        
        # The problem constraints ensure a solution exists, so this return is just a fallback
        return -1