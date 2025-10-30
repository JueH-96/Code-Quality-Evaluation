from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0

        # Upper bound for the search space.
        limit = max(x, y) * 2 + 10          # +10 – small slack that safely covers "overshoot to divisibility"

        visited = [False] * (limit + 1)
        dq = deque()
        dq.append((x, 0))                   # (current value, steps)
        visited[min(x, limit)] = True       # clamp in case x > limit (shouldn't happen with above definition)

        while dq:
            val, steps = dq.popleft()

            # Generate neighbours -------------------------------------------------
            nxts = []

            # ±1 operations
            if val - 1 >= 0:
                nxts.append(val - 1)
            if val + 1 <= limit:
                nxts.append(val + 1)

            # division operations (only if divisible)
            if val % 5 == 0:
                nxts.append(val // 5)
            if val % 11 == 0:
                nxts.append(val // 11)

            # --------------------------------------------------------------------
            for nxt in nxts:
                if nxt == y:
                    return steps + 1
                if not visited[nxt]:
                    visited[nxt] = True
                    dq.append((nxt, steps + 1))

        # Should never be reached with given constraints
        return -1