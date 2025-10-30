class Solution:
    def waysToReachStair(self, k: int) -> int:
        # memoization dictionary to store results of subproblems
        # Key: (current_stair, current_jump, prev_op_was_down)
        # Value: Number of ways to reach k from this state
        memo = {}

        # Max jump exponent we need to consider. 
        # 2^31 is ~2 * 10^9, 2^32 is ~4 * 10^9. Since k is up to 10^9,
        # jump values up to 31 are sufficient for relevant 'up' operations.
        MAX_JUMP_EXP = 31 
        
        # Max relevant stair value above k.
        # This is an empirically chosen constant. If current_stair goes too far above k,
        # it becomes very difficult or impossible to return to k with valid operations.
        # A value like 60 is generally a safe upper bound (related to 2 * MAX_JUMP_EXP).
        MAX_STAIR_OFFSET_ABOVE_K = 60

        def solve(current_stair: int, current_jump: int, prev_op_was_down: bool) -> int:
            # Base cases for pruning the search space

            # If Alice goes below stair 0, it's an invalid path.
            if current_stair < 0:
                return 0

            # If current_jump value becomes too large, 2^current_jump will be enormous.
            # This makes it impossible to reach k or return to k if overshot.
            if current_jump > MAX_JUMP_EXP: 
                return 0

            # If current_stair is significantly above k, it's likely impossible to reach k.
            # This pruning is crucial for performance with large k values.
            # E.g., for k=0, current_stair can legitimately go up to 7 (or even higher)
            # but 0 + 60 = 60 is a sufficient upper bound.
            if current_stair > k + MAX_STAIR_OFFSET_ABOVE_K:
                return 0
            
            # Key for memoization. Tuples are hashable and work as dictionary keys.
            state = (current_stair, current_jump, prev_op_was_down)
            if state in memo:
                return memo[state]

            ways = 0
            
            # If Alice is currently on stair k, this is a valid way to reach stair k.
            # The problem counts each time stair k is reached as a "way".
            if current_stair == k:
                ways += 1

            # Option 1: Go up
            # Alice moves to stair `current_stair + 2^current_jump`
            # `jump` becomes `jump + 1`. `prev_op_was_down` becomes False.
            ways += solve(current_stair + (1 << current_jump), current_jump + 1, False)

            # Option 2: Go down
            # Alice moves to stair `current_stair - 1`.
            # This operation cannot be used consecutively: `prev_op_was_down` must be False.
            # This operation cannot be used on stair 0: `current_stair` must be > 0.
            if not prev_op_was_down and current_stair > 0:
                ways += solve(current_stair - 1, current_jump, True)

            # Store the computed result in memoization table
            memo[state] = ways
            return ways

        # Alice starts on stair 1, with initial jump value of 0, 
        # and no previous operation means prev_op_was_down is False.
        return solve(1, 0, False)