from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        queue = deque([x])
        visited = set([x])
        operations = 0
        while queue:
            size = len(queue)
            operations += 1
            for _ in range(size):
                current = queue.popleft()
                # Apply division by 11 if possible
                if current % 11 == 0:
                    next_val = current // 11
                    if next_val == y:
                        return operations
                    if next_val not in visited:
                        queue.append(next_val)
                        visited.add(next_val)
                # Apply division by 5 if possible
                if current % 5 == 0:
                    next_val = current // 5
                    if next_val == y:
                        return operations
                    if next_val not in visited:
                        queue.append(next_val)
                        visited.add(next_val)
                # Apply decrement by 1 if possible
                if current > 1:
                    next_val = current - 1
                    if next_val == y:
                        return operations
                    if next_val not in visited:
                        queue.append(next_val)
                        visited.add(next_val)
                # Apply increment by 1
                next_val = current + 1
                if next_val == y:
                    return operations
                if next_val not in visited:
                    queue.append(next_val)
                    visited.add(next_val)
        return operations