from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        q = deque([(x, 0)]) # (current_value, operations_count)
        visited = {x}
        while q:
            current_x, operations_count = q.popleft()
            if current_x == y:
                return operations_count
            
            next_values = []
            if current_x % 11 == 0:
                next_values.append(current_x // 11)
            if current_x % 5 == 0:
                next_values.append(current_x // 5)
            next_values.append(current_x - 1)
            next_values.append(current_x + 1)
            
            for next_x in next_values:
                if 0 <= next_x <= 20000 and next_x not in visited:
                    visited.add(next_x)
                    q.append((next_x, operations_count + 1))
        return -1 # Should not reach here as per problem statement