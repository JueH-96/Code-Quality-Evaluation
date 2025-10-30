from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if y >= x:
            return y - x
        
        # BFS to find minimum operations when y < x
        queue = deque()
        queue.append((x, 0))  # (current number, steps)
        visited = set()
        visited.add(x)
        
        while queue:
            curr, steps = queue.popleft()
            
            # Decrement by 1
            if curr - 1 >= 0 and (curr - 1) not in visited:
                if curr - 1 == y:
                    return steps + 1
                queue.append((curr - 1, steps + 1))
                visited.add(curr - 1)
            
            # Increment by 1
            if curr + 1 <= 20000 and (curr + 1) not in visited:
                if curr + 1 == y:
                    return steps + 1
                queue.append((curr + 1, steps + 1))
                visited.add(curr + 1)
            
            # Divide by 5 if divisible
            if curr % 5 == 0:
                next_num = curr // 5
                if 0 <= next_num <= 20000 and next_num not in visited:
                    if next_num == y:
                        return steps + 1
                    queue.append((next_num, steps + 1))
                    visited.add(next_num)
            
            # Divide by 11 if divisible
            if curr % 11 == 0:
                next_num = curr // 11
                if 0 <= next_num <= 20000 and next_num not in visited:
                    if next_num == y:
                        return steps + 1
                    queue.append((next_num, steps + 1))
                    visited.add(next_num)
        
        # Should never reach here as we can always reach y by decrementing or other operations
        return abs(x - y)