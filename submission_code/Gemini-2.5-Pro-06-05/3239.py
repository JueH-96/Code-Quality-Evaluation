import collections

class Solution:
  def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
    """
    Calculates the minimum number of operations to make x equal to y.

    This problem can be modeled as finding the shortest path in a graph where
    numbers are nodes and the allowed operations are edges. Since all operations
    have a weight of 1, Breadth-First Search (BFS) is the ideal algorithm.
    """

    # If x is already less than or equal to y, the optimal path is to
    # simply increment x until it equals y. Any other operation would move
    # x further from the target.
    if x <= y:
        return y - x

    # Initialize a queue for BFS with the starting number and 0 operations.
    queue = collections.deque([(x, 0)])
    
    # Use a set to keep track of visited numbers to avoid redundant work and cycles.
    visited = {x}
    
    while queue:
        current_x, ops = queue.popleft()
        
        # Generate all possible next states from the current number.
        neighbors = []
        if current_x % 11 == 0:
            neighbors.append(current_x // 11)
        if current_x % 5 == 0:
            neighbors.append(current_x // 5)
        neighbors.append(current_x - 1)
        neighbors.append(current_x + 1)
        
        for next_x in neighbors:
            # If we reach the target, we have found the shortest path.
            if next_x == y:
                return ops + 1
            
            # If the neighbor has not been visited, add it to the queue and visited set.
            # We only consider positive integers as per the problem constraints.
            if next_x > 0 and next_x not in visited:
                visited.add(next_x)
                queue.append((next_x, ops + 1))
                
    return -1 # Should be unreachable given the problem constraints