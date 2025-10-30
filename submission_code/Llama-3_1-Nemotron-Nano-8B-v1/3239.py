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
            next_steps = []
            # Decrement
            next_num = current - 1
            if next_num not in visited:
                next_steps.append((next_num, steps + 1))
                visited.add(next_num)
            # Increment
            next_num = current + 1
            if next_num not in visited:
                next_steps.append((next_num, steps + 1))
                visited.add(next_num)
            # Divide by 5
            if current % 5 == 0:
                next_num = current // 5
                if next_num not in visited:
                    next_steps.append((next_num, steps + 1))
                    visited.add(next_num)
            # Divide by 11
            if current % 11 == 0:
                next_num = current // 11
                if next_num not in visited:
                    next_steps.append((next_num, steps + 1))
                    visited.add(next_num)
            # Check if any next step is the target
            for num, s in next_steps:
                if num == y:
                    return s
                queue.append((num, s))
        return -1  # This line is theoretically unreachable as per problem constraints