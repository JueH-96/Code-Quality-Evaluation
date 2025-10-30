from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        # We allow states in [1 .. limit]
        # Bound set large enough to accommodate necessary increments for divisibility.
        limit = 2 * max(x, y) + 10
        
        visited = [False] * (limit + 1)
        q = deque()
        q.append((x, 0))
        visited[x] = True
        
        while q:
            cur, steps = q.popleft()
            # If reached target, return steps
            if cur == y:
                return steps
            
            # generate neighbors
            # 1) divide by 11
            if cur % 11 == 0:
                nxt = cur // 11
                if 1 <= nxt <= limit and not visited[nxt]:
                    visited[nxt] = True
                    q.append((nxt, steps + 1))
            
            # 2) divide by 5
            if cur % 5 == 0:
                nxt = cur // 5
                if 1 <= nxt <= limit and not visited[nxt]:
                    visited[nxt] = True
                    q.append((nxt, steps + 1))
            
            # 3) decrement by 1
            nxt = cur - 1
            if nxt >= 1 and not visited[nxt]:
                visited[nxt] = True
                q.append((nxt, steps + 1))
            
            # 4) increment by 1
            nxt = cur + 1
            if nxt <= limit and not visited[nxt]:
                visited[nxt] = True
                q.append((nxt, steps + 1))
        
        # In principle, BFS should always find y
        return -1