class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque

        if x <= y:
            return y - x

        visited = set()
        queue = deque([(x, 0)])

        while queue:
            current, steps = queue.popleft()

            if current == y:
                return steps

            if current in visited:
                continue

            visited.add(current)

            # Increment
            if current + 1 not in visited:
                queue.append((current + 1, steps + 1))

            # Decrement
            if current - 1 not in visited:
                queue.append((current - 1, steps + 1))

            # Divide by 11
            if current % 11 == 0 and current // 11 not in visited:
                queue.append((current // 11, steps + 1))

            # Divide by 5
            if current % 5 == 0 and current // 5 not in visited:
                queue.append((current // 5, steps + 1))

        return -1  # This line should never be reached given the constraints