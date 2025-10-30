class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        # If x is already equal to y, no operations are needed
        if x == y:
            return 0
        
        # We will use BFS to find the shortest path (minimum operations)
        # from x to y using the allowed operations.
        # States are integers in [1..MAX_VALUE], where MAX_VALUE is chosen
        # so that we don't miss possible optimal paths that involve increments.
        MAX_VALUE = 20000
        
        visited = [False] * (MAX_VALUE + 1)
        queue = deque()
        
        # Start from x with distance 0
        queue.append((x, 0))
        visited[x] = True
        
        while queue:
            current, steps = queue.popleft()
            
            # Try all 4 possible operations:
            # 1) current // 11 if divisible by 11
            if current % 11 == 0:
                nxt = current // 11
                if nxt == y:
                    return steps + 1
                if nxt >= 1 and not visited[nxt]:
                    visited[nxt] = True
                    queue.append((nxt, steps + 1))
            
            # 2) current // 5 if divisible by 5
            if current % 5 == 0:
                nxt = current // 5
                if nxt == y:
                    return steps + 1
                if nxt >= 1 and not visited[nxt]:
                    visited[nxt] = True
                    queue.append((nxt, steps + 1))
            
            # 3) current - 1 (if possible)
            if current > 1:  # remain positive
                nxt = current - 1
                if nxt == y:
                    return steps + 1
                if nxt >= 1 and not visited[nxt]:
                    visited[nxt] = True
                    queue.append((nxt, steps + 1))
            
            # 4) current + 1 (up to MAX_VALUE)
            if current < MAX_VALUE:
                nxt = current + 1
                if nxt == y:
                    return steps + 1
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append((nxt, steps + 1))
        
        # We should always be able to reach y, so this return
        # is just a fallback. 
        return -1