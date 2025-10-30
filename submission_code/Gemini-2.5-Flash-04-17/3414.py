from collections import deque

class Solution:
    def waysToReachStair(self, k: int) -> int:
        # State: (current_stair, num_up_ops_done, can_down)
        # current_stair: Alice's current stair.
        # num_up_ops_done: The number of 'Up' operations Alice has performed so far.
        #                  This means the jump power for the next 'Up' operation is 2^num_up_ops_done.
        # can_down: True if the last operation was NOT Down (including the initial state), False otherwise.

        # Initial state: Alice starts on stair 1, with jump value 0. No operations done yet.
        # Initial state: (stair=1, num_up_ops_done=0, can_down=True).
        initial_state = (1, 0, True)
        
        # counts[(stair, num_up, can_down)] = number of distinct sequences of operations
        #                                     from the start state that result in this state.
        counts = {initial_state: 1}
        
        # Queue for BFS. Stores states to explore.
        q = deque([initial_state])
        
        # Total number of ways to reach stair k.
        total_ways = 0

        # Limit the number of 'Up' operations completed (num_up).
        # If num_up becomes very high, the step size 2^num_up becomes huge.
        # k <= 10^9. log2(10^9) is approximately 29.89.
        # If we use jump power 2^30 (num_up = 30), the step is > 10^9.
        # If we use jump power 2^32 (num_up = 32), the step is > 4 * 10^9.
        # If current stair is already far above k and jump power is very high,
        # subsequent Up operations will move Alice even further away, making it practically
        # impossible to reach k by only using a limited number of future Down operations.
        # We can prune the 'Up' branch based on `num_up`.
        # Let's explore states where `num_up_ops_done` is up to 32.
        # If num_up_ops_done is 32, the next Up op uses 2^32.
        # If num_up_ops_done becomes 33, we stop the Up branch from that state.
        # The condition `num_up <= max_num_up_ops` allows states with num_up up to 32.
        # The Up operation increments num_up, potentially reaching 33.
        max_num_up_ops = 32

        while q:
            stair, num_up, can_down = q.popleft()
            
            # Get the number of ways to reach this state.
            # We use pop to remove the state from the dictionary once processed,
            # effectively marking it as visited and preventing re-processing from this exact state.
            # If other paths lead to this state later, their counts will be added, and the state
            # will be added back to the queue.
            count = counts.pop((stair, num_up, can_down))

            # If the current stair is k, any path reaching this state is a valid way to reach k.
            # Add the number of ways to reach this state to the total count.
            if stair == k:
                total_ways += count

            # Option 1: Go Down (stair - 1)
            # Condition: Last operation was NOT Down (`can_down` is True) and current stair is > 0.
            if can_down and stair > 0:
                next_stair_down = stair - 1
                next_num_up_down = num_up # Number of up ops done doesn't change
                next_can_down_down = False # Last operation was Down
                next_state_down = (next_stair_down, next_num_up_down, next_can_down_down)
                
                # Add the number of ways to reach the current state to the next state's count.
                counts[next_state_down] = counts.get(next_state_down, 0) + count
                
                # Add the next state to the queue to explore later.
                # Adding unconditionally is fine; the counts dictionary handles accumulation,
                # and popping from the queue processes each unique state (for a given count sum at that moment).
                q.append(next_state_down)

            # Option 2: Go Up (stair + 2^num_up)
            # Condition for performing the Up operation:
            # 1. The number of completed Up operations so far (`num_up`) plus the current Up operation
            #    should not result in exceeding the practical limit on `num_up`.
            #    The next state will have `next_num_up_up = num_up + 1`.
            #    So, we allow the Up operation if the current `num_up` is <= `max_num_up_ops`.
            # 2. Pruning: If the current number of completed Up operations (`num_up`) is already high (e.g., 32)
            #    AND the current stair is significantly above `k` (e.g., > k + 1),
            #    then performing another large jump will likely move Alice even further away,
            #    making it impossible to reach `k` with a limited number of subsequent Down operations.
            #    If `num_up >= 32` and `stair > k + 1`, we prune the Up branch.
            #    The condition to perform Up is: `num_up <= max_num_up_ops` AND NOT (`num_up >= 32` AND `stair > k + 1`).
            #    This simplifies to: `num_up < 32` OR (`num_up == 32` AND `stair <= k + 1`).

            perform_up = False
            if num_up <= max_num_up_ops: # Check condition 1: limit on num_up
                 # Check condition 2: pruning for large num_up and large stair
                 if num_up < 32 or stair <= k + 1:
                     perform_up = True

            if perform_up:
                step_size = 1 << num_up # Calculate the jump step (2^num_up)
                next_stair_up = stair + step_size
                next_num_up_up = num_up + 1 # Increment number of completed Up ops
                next_can_down_up = True # Last operation was Up
                next_state_up = (next_stair_up, next_num_up_up, next_can_down_up)
                
                # Add the number of ways to reach the current state to the next state's count.
                counts[next_state_up] = counts.get(next_state_up, 0) + count
                
                # Add the next state to the queue.
                q.append(next_state_up)

        # After the BFS completes, total_ways contains the sum of counts for all states (stair, num_up, can_down)
        # where stair == k that were reached.
        return total_ways