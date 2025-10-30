import functools

class Solution:
  def waysToReachStair(self, k: int) -> int:
    # Using a dictionary for memoization
    memo = {}

    # Maximum jump power index to consider. 2^30 is ~10^9.
    # A jump_idx slightly larger than k's bit length should be enough.
    # e.g., k.bit_length() + C where C is a small constant (like 4 or 5).
    # For k up to 10^9, k.bit_length() is at most 30. So, MAX_JUMP_POWER ~35 is reasonable.
    MAX_JUMP_POWER = 35 

    def solve(current_stair: int, current_jump_idx: int, prev_op_was_down: bool) -> int:
      state = (current_stair, current_jump_idx, prev_op_was_down)
      if state in memo:
        return memo[state]

      # Pruning condition 1: If current_jump_idx is too high.
      if current_jump_idx >= MAX_JUMP_POWER:
        return 0
      
      # Pruning condition 2: If current_stair is too far above k.
      # If current_stair > k+1, it's generally impossible to reach k again.
      # This is because from k+1, one can go down to k.
      # From k+2, one can go down to k+1. If prev_op_was_down is now True,
      # the next operation must be "up", moving from k+1 to k+1 + 2^jump_val, further from k.
      # The value of current_stair == k is counted before this pruning.
      if current_stair > k + 1:
          memo[state] = 0
          return 0

      # Initialize count for this state. If current_stair is k, this state itself is one way to reach k.
      res = 0
      if current_stair == k:
        res += 1

      # Option 1: Go down
      # Can go down if not on stair 0 and previous operation was not "go down".
      if not prev_op_was_down:
        if current_stair > 0: 
          res += solve(current_stair - 1, current_jump_idx, True)
      
      # Option 2: Go up
      # The next stair is current_stair + 2^current_jump_idx.
      # The jump power index for the subsequent jump will be current_jump_idx + 1.
      res += solve(current_stair + (1 << current_jump_idx), current_jump_idx + 1, False)
      
      memo[state] = res
      return res

    # Initial call: Alice starts at stair 1, current_jump_idx = 0 (for 2^0), prev_op_was_down = False.
    return solve(1, 0, False)