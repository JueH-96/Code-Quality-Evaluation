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
            next_states = []
            if current % 11 == 0:
                next_states.append(current // 11)
            if current % 5 == 0:
                next_states.append(current // 5)
            next_states.append(current - 1)
            next_states.append(current + 1)
            for state in next_states:
                if state not in visited and state >= 0 and state <= 10000:
                    visited.add(state)
                    queue.append((state, steps + 1))
        return -1  # Should not reach here as per constraints