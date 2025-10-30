class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        # If already equal, no operations are required.
        if x == y:
            return 0
        
        # Define safe boundaries for our search:
        # We keep states in the range [1, upper_bound].
        # Because x, y <= 10^4, we choose an upper bound that is safely above max(x, y).
        upper_bound = 50000
        
        # Use a deque for BFS. Each element is a tuple (current_value, operations_count).
        queue = deque([(x, 0)])
        visited = set([x])
        
        while queue:
            current, ops = queue.popleft()
            
            # Try all possible moves:
            # 1. Division by 11 if possible.
            if current % 11 == 0:
                nxt = current // 11
                if nxt == y:
                    return ops + 1
                if nxt >= 1 and nxt <= upper_bound and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, ops + 1))
                    
            # 2. Division by 5 if possible.
            if current % 5 == 0:
                nxt = current // 5
                if nxt == y:
                    return ops + 1
                if nxt >= 1 and nxt <= upper_bound and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, ops + 1))
            
            # 3. Decrement by 1.
            nxt = current - 1
            if nxt == y:
                return ops + 1
            if nxt >= 1 and nxt <= upper_bound and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, ops + 1))
            
            # 4. Increment by 1.
            nxt = current + 1
            if nxt == y:
                return ops + 1
            if nxt >= 1 and nxt <= upper_bound and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, ops + 1))
                
        # If the while loop ends without finding y (which theoretically should not happen),
        # return -1 to indicate failure.
        return -1