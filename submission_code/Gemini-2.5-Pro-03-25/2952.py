import sys 
from typing import List

class Solution:
    """
    Solves the minimum time problem using dynamic programming.
    The problem asks for the minimum time `t` such that the sum of elements in `nums1` can be made less than or equal to `x`.
    At each second `s` (from 1 to `t`), `nums1[i]` increases by `nums2[i]` for all `i`. After the increments, we can choose *one* index `k` and set `nums1[k] = 0`.

    Let's analyze the state of `nums1` at the end of time `t`.
    If no zeroing operations were performed on index `i`, its value would be `nums1[i] + t * nums2[i]`.
    If index `i` was chosen for the zeroing operation at least once, let the last time it was chosen be second `s_{i, max}` (where `1 <= s_{i, max} <= t`).
    Then its value at the end of time `t` will be `(t - s_{i, max}) * nums2[i]`. Note that if `s_{i, max} = t`, the value is 0.

    The total sum at time `t`, `Sum(t)`, can be expressed as the sum without operations minus the total reduction achieved.
    The sum without operations is `Sum_base(t) = sum(nums1) + t * sum(nums2)`.
    The reduction achieved for an index `i` that was chosen (last time at `s_{i, max}`) is the difference between its value without operation and its final value:
    Reduction(i) = `(nums1[i] + t * nums2[i]) - (t - s_{i, max}) * nums2[i]`
                 = `nums1[i] + t * nums2[i] - t * nums2[i] + s_{i, max} * nums2[i]`
                 = `nums1[i] + s_{i, max} * nums2[i]`.
    If index `i` was never chosen, the reduction is 0.
    
    Let $I_{chosen}$ be the set of indices chosen at least once among the $t$ operations. Let $k_1, \dots, k_t$ be the sequence of chosen indices.
    The total reduction $R(t)$ is $\sum_{i \in I_{chosen}} (nums1[i] + s_{i, max} * nums2[i])$.
    We want to choose the sequence $k_1, \dots, k_t$ to maximize $R(t)$.
    The final sum is `Sum(t) = Sum_base(t) - R_{max}(t)`. We need the minimum `t` such that `Sum(t) <= x`.

    This maximization problem can be solved using dynamic programming, resembling a variation of the knapsack problem.
    Let `dp[j]` store the maximum possible value of $\sum_{p \in P} (nums1[p] + nums2[p] * s_{p, max})$ using exactly `j` operations (i.e., at time `t=j`), where `P` is the set of indices operated on.
    It turns out that sorting the elements based on `nums2[i]` allows for an efficient DP calculation.
    Let pairs be `(nums2[i], nums1[i])`. Sort these pairs based on `nums2[i]`.
    The DP state `dp[j]` will represent the maximum value of the sum $\sum_{p \in P} (a_p + b_p \times \text{rank}_p)$ for a chosen set $P$ of size $j$. Here $a_p = nums1[p]$, $b_p = nums2[p]$, and $\text{rank}_p$ refers to the assigned second index (from 1 to $j$). The DP transition effectively finds the optimal assignment.
    """
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        """
        Calculates the minimum time to make the sum of nums1 <= x.

        Args:
          nums1: The first list of integers.
          nums2: The second list of integers, representing increments per second.
          x: The target maximum sum.

        Returns:
          The minimum time required, or -1 if impossible.
        """
        
        n = len(nums1)
        
        # Calculate initial sum of nums1 and sum of nums2
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        
        # Base case: If initial sum is already less than or equal to x, no time is needed.
        if sum1 <= x:
            return 0
            
        # Create pairs of (nums2[i], nums1[i]) to facilitate DP based on sorted nums2 values.
        # Let b = nums2[i], a = nums1[i].
        pairs = []
        for i in range(n):
            pairs.append((nums2[i], nums1[i]))
            
        # Sort pairs based on nums2[i] (the 'b' value). This order is crucial for the DP state logic.
        # Items with smaller 'b' are processed first.
        pairs.sort()
        
        # Initialize DP array. dp[j] will store the maximum possible reduction achievable using exactly j operations.
        # The reduction is defined as sum(a_p + s_{p, max} * b_p) over the chosen set P of indices.
        dp = [0] * (n + 1)  # dp array size n+1 for 0..n operations
        
        # Fill the DP table using the sorted pairs
        # Outer loop iterates through each item (pair) from the sorted list.
        for i in range(n):
            # Current item properties: b = nums2 value, a = nums1 value
            b, a = pairs[i]
            
            # Inner loop updates DP states. We iterate downwards for j from i+1 to 1.
            # This ensures that each item is considered at most once for a fixed number of operations 'j'.
            # 'j' represents the total number of operations (and thus corresponds to time 't' = j).
            # The maximum number of items considered up to outer loop `i` is `i+1`.
            for j in range(i + 1, 0, -1):
                 # The DP transition: dp[j] = max(current dp[j], value if we include item i).
                 # The value if we include item i is based on the state dp[j-1] (max reduction using j-1 items from previously considered items)
                 # plus the contribution of the current item. The contribution is 'a + b * j'.
                 # This 'a + b * j' term corresponds to the reduction `nums1[p] + nums2[p] * s_{p, max}` where `s_{p, max}` is effectively assigned the value `j`.
                 # The sorting by `b` and this DP structure correctly computes the maximum reduction achievable for `j` operations.
                 dp[j] = max(dp[j], dp[j-1] + a + b * j)

        # After filling the DP table, check for the minimum time 't'.
        # Time 't' corresponds to using 't' operations.
        for t in range(1, n + 1):
            # Calculate the total sum at time 't' assuming no operations were performed.
            current_total_sum = sum1 + sum2 * t
            
            # The maximum reduction achievable with 't' operations is stored in dp[t].
            max_reduction = dp[t]
            
            # Calculate the minimum possible sum at time 't' by applying the optimal 't' operations.
            min_achievable_sum = current_total_sum - max_reduction
            
            # If this minimum achievable sum is less than or equal to x, we found the minimum time 't'.
            if min_achievable_sum <= x:
                return t
                
        # If the loop finishes without finding a suitable time t (from 1 to n), 
        # it means the sum cannot be reduced to x or below even using operations on all n elements.
        return -1