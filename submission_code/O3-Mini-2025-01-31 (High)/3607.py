from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # We'll precompute the smallest prime factor (spf) for all numbers up to max(nums).
        # For a composite number x, its greatest proper divisor is x // (smallest factor)
        # so the result of “dividing by the greatest proper divisor” is:
        #      new value = x / (x // spf(x)) = spf(x)
        #
        # For prime numbers (and 1), the only proper divisor is 1 so operating on them
        # would yield the same number. Thus it never makes sense to do that.
        #
        # So for every number x, we have two “options”:
        #    • Option 0: Leave it as is. (cost = 0)
        #    • Option 1: If x is composite (i.e. x > 1 and spf(x) < x), then operate on it so that
        #                it becomes spf(x) (cost = 1).
        #
        # We want to choose one option at each index so that the final array is non-decreasing,
        # and we want the minimum total cost.

        n = len(nums)
        if n == 0:
            return 0
        
        max_val = max(nums)
        spf = list(range(max_val + 1))
        # Basic sieve to compute spf values.
        for i in range(2, int(max_val ** 0.5) + 1):
            if spf[i] == i:  # i is prime.
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        # Helper: given a number x, return its "operated value" (x divided by its greatest proper divisor)
        # if x is composite. Otherwise, if x is prime or 1 (so that no beneficial operation exists), return None.
        def operated_value(x: int) -> int:
            if x > 1 and spf[x] < x:
                return spf[x]
            return None

        INF = float('inf')
        
        # We use dynamic programming where for each element we keep two possible states:
        # state0: we do not operate on the current number (value remains = x, cost addition = 0)
        # state1: if available (i.e. if x is composite), we operate so that value becomes spf(x) (cost addition = 1)
        #
        # In our DP recurrence the ordering condition is enforced between consecutive indices.
        # That is, if the previous element (whichever state we picked) ended with final value y,
        # then for the current candidate value v (either x or spf(x)) we require y <= v.
        #
        # We keep for the “previous index” two states:
        #   dp0, with final value state0_val (if we did nothing at that index)
        #   dp1, with final value state1_val (if we operated)
        #
        # Initialize for index 0:
        x = nums[0]
        dp0 = 0               # leave it as is → value = x.
        state0_val = x
        op_val = operated_value(x)
        if op_val is not None:
            dp1 = 1           # cost 1 if we operate on it
            state1_val = op_val
        else:
            dp1 = INF
            state1_val = None
        
        # Process the rest of the array
        for i in range(1, n):
            x = nums[i]
            # Option if we do nothing: candidate value = x, cost add = 0.
            cur0 = x
            # Option if we operate (if available): candidate value = spf(x) (which is < x), cost add = 1.
            cur1 = operated_value(x)
            
            new_dp0 = INF  # new cost if we decide not to operate on nums[i]
            new_dp1 = INF  # new cost if we do operate  (if available)
            
            # For Option 0 at index i: we want the previous state's final value to be <= cur0.
            if state0_val is not None and state0_val <= cur0:
                new_dp0 = min(new_dp0, dp0)
            if state1_val is not None and state1_val <= cur0:
                new_dp0 = min(new_dp0, dp1)
            
            # For Option 1 at index i: if operation is available (cur1 is not None),
            # we add 1 extra cost and require previous state's final value <= cur1.
            if cur1 is not None:
                if state0_val is not None and state0_val <= cur1:
                    new_dp1 = min(new_dp1, dp0 + 1)
                if state1_val is not None and state1_val <= cur1:
                    new_dp1 = min(new_dp1, dp1 + 1)
            
            # If neither option is possible, then it is impossible to get a non-decreasing array.
            if new_dp0 == INF and new_dp1 == INF:
                return -1
            
            # Update state for the current index:
            dp0, state0_val = new_dp0, cur0
            if new_dp1 == INF:
                dp1, state1_val = INF, None
            else:
                dp1, state1_val = new_dp1, cur1

        ans = min(dp0, dp1)
        return ans if ans != INF else -1

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minOperations([25,7]))   # Expected output: 1 
    # Explanation: operate on 25 to get 5 so array becomes [5, 7].
    
    # Example 2:
    print(sol.minOperations([7,7,6]))   # Expected output: -1
    
    # Example 3:
    print(sol.minOperations([1,1,1,1])) # Expected output: 0