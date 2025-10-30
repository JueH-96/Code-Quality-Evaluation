from collections import deque

def minimumOperationsToMakeEqual(x: int, y: int) -> int:
    if x == y:
        return 0
    queue = deque()
    queue.append((x, 0))
    visited = set()
    visited.add(x)
    
    while queue:
        current, steps = queue.popleft()
        
        # Apply all four operations
        # Divide by 5
        if current % 5 == 0:
            next_num = current // 5
            if next_num not in visited:
                if next_num == y:
                    return steps + 1
                visited.add(next_num)
                queue.append((next_num, steps + 1))
        # Divide by 11
        if current % 11 == 0:
            next_num = current // 11
            if next_num not in visited:
                if next_num == y:
                    return steps + 1
                visited.add(next_num)
                queue.append((next_num, steps + 1))
        # Decrement
        next_num = current - 1
        if next_num not in visited:
            if next_num == y:
                return steps + 1
            visited.add(next_num)
            queue.append((next_num, steps + 1))
        # Increment
        next_num = current + 1
        if next_num not in visited:
            if next_num == y:
                return steps + 1
            visited.add(next_num)
            queue.append((next_num, steps + 1))
    
    # This return is theoretically unreachable as per problem constraints
    return -1