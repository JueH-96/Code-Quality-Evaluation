from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0

        queue = deque([(x, 0)])
        visited = {x}

        while queue:
            current_x, operations = queue.popleft()

            # Operation 1: Divide by 11
            if current_x % 11 == 0:
                next_x = current_x // 11
                if next_x == y:
                    return operations + 1
                if next_x > 0 and next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))

            # Operation 2: Divide by 5
            if current_x % 5 == 0:
                next_x = current_x // 5
                if next_x == y:
                    return operations + 1
                if next_x > 0 and next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))

            # Operation 3: Decrement by 1
            next_x = current_x - 1
            if next_x > 0:
                if next_x == y:
                    return operations + 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))

            # Operation 4: Increment by 1
            next_x = current_x + 1
            # While there isn't a strict upper bound, we can limit the search space
            # based on the problem constraints. Since we are aiming for y, exploring
            # much beyond a reasonable range around y might be inefficient.
            # However, BFS will naturally find the shortest path.
            # Let's consider a practical upper limit for exploration.
            if next_x <= 2 * 10**4 + 5:  # Heuristic upper bound
                if next_x == y:
                    return operations + 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))

        return -1 # Should not happen given the problem constraints