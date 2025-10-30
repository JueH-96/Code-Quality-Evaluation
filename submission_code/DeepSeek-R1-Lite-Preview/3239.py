from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0  # No operations needed if x is already equal to y
        
        # Initialize the queue with the starting value of x and 0 operations
        queue = deque()
        queue.append((x, 0))
        
        # Set to keep track of visited x values to avoid redundant computations
        visited = set()
        visited.add(x)
        
        while queue:
            current_x, operations = queue.popleft()
            
            # Operation 1: Divide by 11 if divisible
            if current_x % 11 == 0:
                next_x = current_x // 11
                if next_x == y:
                    return operations + 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))
            
            # Operation 2: Divide by 5 if divisible
            if current_x % 5 == 0:
                next_x = current_x // 5
                if next_x == y:
                    return operations + 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))
            
            # Operation 3: Decrement by 1 if current_x > 1
            if current_x > 1:
                next_x = current_x - 1
                if next_x == y:
                    return operations + 1
                if next_x not in visited:
                    visited.add(next_x)
                    queue.append((next_x, operations + 1))
            
            # Operation 4: Increment by 1
            next_x = current_x + 1
            if next_x == y:
                return operations + 1
            if next_x not in visited:
                visited.add(next_x)
                queue.append((next_x, operations + 1))
        
        # If y is not reachable, though per problem constraints, it should be.
        return -1