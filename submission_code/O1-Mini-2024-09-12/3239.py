from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        max_limit = 2 * max(x, y) + 10
        visited = set()
        queue = deque()
        queue.append((x, 0))
        while queue:
            current, steps = queue.popleft()
            if current == y:
                return steps
            if current in visited or current < 1 or current > max_limit:
                continue
            visited.add(current)
            # Decrement
            queue.append((current - 1, steps + 1))
            # Increment
            queue.append((current + 1, steps + 1))
            # Divide by 5
            if current % 5 == 0:
                queue.append((current // 5, steps + 1))
            # Divide by 11
            if current % 11 == 0:
                queue.append((current // 11, steps + 1))
        return -1