class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque

        max_limit = 20000  # Assuming y can be up to 10000, we can double it to be safe
        visited = set()
        queue = deque()
        queue.append((x, 0))
        visited.add(x)

        while queue:
            curr_x, steps = queue.popleft()
            if curr_x == y:
                return steps
            next_steps = steps + 1
            # Operation 1: Divide x by 11 if x is a multiple of 11
            if curr_x % 11 == 0:
                next_x = curr_x // 11
                if next_x not in visited and 1 <= next_x <= max_limit:
                    visited.add(next_x)
                    queue.append((next_x, next_steps))
            # Operation 2: Divide x by 5 if x is a multiple of 5
            if curr_x % 5 == 0:
                next_x = curr_x // 5
                if next_x not in visited and 1 <= next_x <= max_limit:
                    visited.add(next_x)
                    queue.append((next_x, next_steps))
            # Operation 3: Decrement x by 1
            if curr_x - 1 >= 1:
                next_x = curr_x - 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, next_steps))
            # Operation 4: Increment x by 1
            if curr_x + 1 <= max_limit:
                next_x = curr_x + 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, next_steps))

        # If we reach here, then y is not reachable from x (should not happen given constraints)
        return -1