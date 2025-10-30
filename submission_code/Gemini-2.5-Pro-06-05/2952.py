from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        """
        Finds the minimum time to make the sum of nums1 <= x.
        """
        n = len(nums1)
        
        # Create pairs of (nums1[i], nums2[i]) and sort them by nums2[i].
        # This sorting is essential for the DP state transition logic.
        pairs = sorted(zip(nums1, nums2), key=lambda p: p[1])
        
        # dp[k] will store the maximum "benefit" we can get by performing k operations.
        # The benefit for choosing a set of k items is defined as:
        # sum(nums1_chosen) + sum(time * nums2_chosen), where times are 1..k assigned
        # to items sorted by their nums2 values.
        dp = [0] * (n + 1)
        
        # DP to calculate max benefit for each number of operations k.
        # Outer loop `i` iterates through items from the sorted list.
        for i in range(1, n + 1):
            a, b = pairs[i - 1]  # Current item's nums1 and nums2 values
            
            # Inner loop `k` iterates downwards for the number of operations.
            # This is the standard space optimization for this type of DP,
            # allowing us to use values from the (i-1)-th state correctly.
            for k in range(i, 0, -1):
                # We have two choices for the i-th item:
                # 1. Don't include it: The max benefit for k items remains dp[k] (from the previous i-1 state).
                # 2. Include it: We must have chosen k-1 items from the first i-1 items.
                #    The benefit from this new item is `a + k * b`, as it's operated on at time k.
                #    The total benefit is `dp[k-1] (from i-1 state) + a + k * b`.
                dp[k] = max(dp[k], dp[k - 1] + a + k * b)
                
        # Pre-calculate the initial sum of nums1 and the sum of increments per second.
        s1 = sum(nums1)
        s2 = sum(nums2)
        
        # Check for each possible time `k` from 0 to n.
        # For k operations, the optimal time is k seconds.
        for k in range(n + 1):
            # Total sum at time k = (base sum without ops) - (max benefit from k ops)
            total_sum = s1 + k * s2 - dp[k]
            
            if total_sum <= x:
                return k
                
        # If no time `k` satisfies the condition, it's impossible.
        return -1