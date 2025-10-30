from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0

        queue = deque([(x, 0)])  # (current value of x, number of operations)
        visited = set([x])

        while queue:
            current_x, operations = queue.popleft()

            # Try all possible operations
            next_states = []
            if current_x % 11 == 0:
                next_states.append(current_x // 11)
            if current_x % 5 == 0:
                next_states.append(current_x // 5)
            next_states.append(current_x - 1)
            next_states.append(current_x + 1)

            for next_x in next_states:
                if next_x == y:
                    return operations + 1
                if next_x not in visited and 1 <= next_x <= 10**4:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))

        return -1  # This line should never be reached due to the problem constraints