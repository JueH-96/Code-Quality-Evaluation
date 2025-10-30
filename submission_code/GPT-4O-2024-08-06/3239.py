class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque

        # BFS approach to find the minimum number of operations
        queue = deque([(x, 0)])  # (current value of x, number of operations)
        visited = set([x])  # to avoid revisiting the same state

        while queue:
            current, ops = queue.popleft()

            if current == y:
                return ops

            # Generate possible next states
            next_states = []

            if current % 11 == 0:
                next_states.append(current // 11)
            if current % 5 == 0:
                next_states.append(current // 5)

            next_states.append(current - 1)
            next_states.append(current + 1)

            for next_state in next_states:
                if next_state not in visited and 1 <= next_state <= 10000:
                    visited.add(next_state)
                    queue.append((next_state, ops + 1))

        return -1  # This should never be reached due to constraints