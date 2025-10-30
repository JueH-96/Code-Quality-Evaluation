class Solution:
  def valueAfterKSeconds(self, n: int, k: int) -> int:
    MOD = 10**9 + 7
    
    # dp array representing the current state 'a'.
    # dp[i] stores a[i].
    # Initial state a_0 (after 0 seconds): all elements are 1.
    dp = [1] * n
    
    # We need to simulate k seconds.
    # The outer loop runs k times. Each iteration computes the state for the next second.
    # Initial `dp` is a_0.
    # After 1st iteration of outer loop, `dp` becomes a_1.
    # ...
    # After k-th iteration of outer loop, `dp` becomes a_k.
    
    for _ in range(k): # This loop runs k times.
      # The update rule is: a_new[i] = sum(a_old[j] for j from 0 to i).
      # This is equivalent to a_new being the prefix sum array of a_old.
      # The recurrence for this prefix sum is:
      #   a_new[0] = a_old[0]
      #   a_new[i] = a_old[i] + a_new[i-1]  (for i > 0)
      #
      # The in-place update `dp[j] = (dp[j] + dp[j-1]) % MOD` achieves this:
      # - `dp[j]` on the right-hand side is `a_old[j]`.
      # - `dp[j-1]` on the right-hand side is `a_new[j-1]` (already updated in this pass).
      # So, `dp_new[j]` correctly becomes `(dp_old[j] + dp_new[j-1]) % MOD`.
      #
      # `dp[0]` is correctly handled: it's initialized to 1 and never modified
      # by the inner loop, reflecting that a_s[0] is always 1.
      
      for j in range(1, n):
        dp[j] = (dp[j] + dp[j-1]) % MOD
            
    # After k seconds, the array `dp` stores the state `a_k`.
    # We need to return the value of `a_k[n-1]`.
    return dp[n-1]