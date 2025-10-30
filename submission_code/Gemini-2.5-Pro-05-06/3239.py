import collections

class Solution:
  def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
    if x <= y:
      # If x == y, 0 operations are needed.
      # If x < y, as proven in the thought process, the optimal strategy is to
      # only increment x. So, y - x operations.
      return y - x

    # Case: x > y
    # Use Breadth-First Search (BFS)
    # The queue stores tuples of (current_value, num_operations)
    queue = collections.deque([(x, 0)])
    # The visited set stores values that have been added to the queue,
    # to prevent cycles and redundant computations.
    visited = {x}

    while queue:
      curr_val, curr_ops = queue.popleft()

      if curr_val == y:
        return curr_ops

      # Order of exploring neighbors can affect performance slightly, but not correctness.
      # Prioritizing division as it reduces numbers faster.

      # Option 1: Divide by 11
      if curr_val % 11 == 0:
        next_val = curr_val // 11
        if next_val not in visited:
          visited.add(next_val)
          queue.append((next_val, curr_ops + 1))
      
      # Option 2: Divide by 5
      if curr_val % 5 == 0:
        next_val = curr_val // 5
        if next_val not in visited:
          visited.add(next_val)
          queue.append((next_val, curr_ops + 1))
          
      # Option 3: Decrement by 1
      next_val_dec = curr_val - 1
      # Values must remain positive (>=1).
      if next_val_dec >= 1: 
        if next_val_dec not in visited:
          visited.add(next_val_dec)
          queue.append((next_val_dec, curr_ops + 1))

      # Option 4: Increment by 1
      # This operation is useful if curr_val+k becomes a multiple of 5 or 11,
      # enabling a division. E.g., 54 -> 55 -> 5.
      # The maximum value reached would be roughly x_initial + 10.
      # Given x <= 10^4, the number of states up to ~10010 is manageable.
      next_val_inc = curr_val + 1
      if next_val_inc not in visited:
        # For this problem, an explicit upper bound on next_val_inc (e.g. x_initial + 25)
        # is not strictly necessary as y < x, and BFS naturally prunes paths
        # that become too long by excessively increasing curr_val.
        # The number of states is limited by the problem constraints on x, y.
        visited.add(next_val_inc)
        queue.append((next_val_inc, curr_ops + 1))
        
    return -1 # Should theoretically not be reached if y is always reachable.