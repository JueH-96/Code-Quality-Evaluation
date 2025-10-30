class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x <= y:
            return y - x

        # BFS approach to find the minimum number of operations
        from collections import deque

        queue = deque([(x, 0)])  # (current_value, number_of_operations)
        visited = set()
        visited.add(x)

        while queue:
            current, steps = queue.popleft()

            if current == y:
                return steps

            # Try all possible operations
            if current % 11 == 0 and current // 11 not in visited:
                queue.append((current // 11, steps + 1))
                visited.add(current // 11)

            if current % 5 == 0 and current // 5 not in visited:
                queue.append((current // 5, steps + 1))
                visited.add(current // 5)

            if current - 1 not in visited:
                queue.append((current - 1, steps + 1))
                visited.add(current - 1)

            if current + 1 not in visited:
                queue.append((current + 1, steps + 1))
                visited.add(current + 1)

        return -1  # This line should never be reached