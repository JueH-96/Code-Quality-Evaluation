from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # BFS with queue
        queue = deque([(x, 0)])
        visited = {x}
        while queue:
            cur, step = queue.popleft()
            if cur == y:
                return step
            for nxt in (cur//11, cur//5, cur-1, cur+1):
                if nxt > 0 and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, step+1))
        return -1