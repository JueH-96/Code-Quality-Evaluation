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
            
            # Generate all possible next numbers
            # Divide by 11 if possible
            if current % 11 == 0:
                next_num = current // 11
                if next_num == y:
                    return steps + 1
                if next_num not in visited:
                    visited.add(next_num)
                    queue.append((next_num, steps + 1))
            # Divide by 5 if possible
            if current % 5 == 0:
                next_num = current // 5
                if next_num == y:
                    return steps + 1
                if next_num not in visited:
                    visited.add(next_num)
                    queue.append((next_num, steps + 1))
            # Decrement by 1
            next_num = current - 1
            if next_num == y:
                return steps + 1
            if next_num not in visited:
                visited.add(next_num)
                queue.append((next_num, steps + 1))
            # Increment by 1
            next_num = current + 1
            if next_num == y:
                return steps + 1
            if next_num not in visited:
                visited.add(next_num)
                queue.append((next_num, steps + 1))
        
        return -1  # This line is theoretically unreachable given problem constraints