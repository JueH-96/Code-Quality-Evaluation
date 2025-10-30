from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # If they're already equal, no operations are needed.
        if x == y:
            return 0

        # We allow a small overshoot above max(x,y) to accommodate
        # the possibility of incrementing into a divisible-by-5 or 11 number.
        max_limit = max(x, y) + 11

        # visited[i] will be True if we've already enqueued i.
        visited = [False] * (max_limit + 1)

        # BFS queue storing (current_value, steps_so_far)
        q = deque([(x, 0)])
        visited[x] = True

        while q:
            curr, steps = q.popleft()

            # Generate all possible next states:
            candidates = []
            if curr % 11 == 0:
                candidates.append(curr // 11)
            if curr % 5 == 0:
                candidates.append(curr // 5)
            candidates.append(curr - 1)
            candidates.append(curr + 1)

            for nxt in candidates:
                # If we've reached y, return the answer immediately.
                if nxt == y:
                    return steps + 1
                # Otherwise, enqueue nxt if it's in bounds and unvisited.
                if 0 <= nxt <= max_limit and not visited[nxt]:
                    visited[nxt] = True
                    q.append((nxt, steps + 1))

        # In theory we should always be able to reach y, but return -1 as a safeguard.
        return -1