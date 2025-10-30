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
            if current == y:
                return steps
            # Generate all possible next states
            # Decrement by 1
            next_val = current - 1
            if next_val >= 1 and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
            # Increment by 1
            next_val = current + 1
            if next_val <= 10**4 and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
            # Divide by 5 if divisible
            if current % 5 == 0:
                next_val = current // 5
                if next_val >= 1 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            # Divide by 11 if divisible
            if current % 11 == 0:
                next_val = current // 11
                if next_val >= 1 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
        return -1  # Should not reach here as per constraints