from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # Create pairs (nums1[i], nums2[i])
        # Sort pairs based on nums2[i] ascending. This ordering is crucial for the DP.
        pairs = sorted(zip(nums1, nums2), key=lambda item: item[1])
        
        # dp[j] will store the maximum possible sum of the values
        # (value_of_element_before_reset_at_time_s + nums2_of_element)
        # accumulated over exactly j operations, when choosing operations optimally.
        # The value added by the j-th operation (at time j) on an element (a, b),
        # assuming it was not reset by the first j-1 operations, is a + (j-1)*b + b = a + b*j.
        dp = [0] * (n + 1)
        
        # Iterate through sorted elements (a, b) where a=nums1[i], b=nums2[i]
        for a, b in pairs:
            # Iterate through possible number of operations j downwards
            # This ensures that when we compute dp[j], dp[j-1] still holds the maximum
            # reduction using j-1 operations considering the elements processed so far
            # *before* considering the current element as the target of the j-th operation.
            for j in range(n, 0, -1):
                # We consider using the j-th operation (at time j) as the last reset
                # for the current element (a, b). If this element was not reset by
                # the previous j-1 operations (which optimally achieved dp[j-1] reduction),
                # its value before reset at time j would be a + (j-1)*b. Resetting it
                # removes this value plus its growth for the current second b, giving
                # a reduction of (a + (j-1)*b) + b = a + b*j compared to the sum
                # just before the operation. This contributes to the total reduction.
                dp[j] = max(dp[j], dp[j-1] + a + b * j)
        
        # Calculate initial sum of nums1 and sum of nums2
        s1 = sum(nums1)
        s2 = sum(nums2)
        
        # The sum of nums1 at time t after t optimal operations is the initial sum
        # plus the total growth minus the total reduction achieved by the operations.
        # Sum(t) = Sum(initial) + t * Sum(nums2) - TotalReduction(t)
        # The maximum total reduction after t operations is given by dp[t].
        # We need to find the minimum time t such that s1 + t * s2 - dp[t] <= x
        
        # Check time t = 0
        if s1 <= x:
            return 0
            
        # Check time t = 1 to n.
        # If the target sum is not achievable by time n, it is not achievable
        # at any time t > n, because the sum function S(t) = s1 + t*s2 - dp[t]
        # does not decrease for t > n if S(n) > x and sum(nums2) >= 0.
        # Specifically, the rate of change S(t) - S(t-1) = s2 - (dp[t] - dp[t-1]).
        # The term dp[t] - dp[t-1] is the maximum possible reduction achieved by the
        # t-th operation (at time t). While this can be large, the DP structure
        # and the fact that we have optimally managed resets up to time n suggest
        # that for t > n, the marginal reduction dp[t] - dp[t-1] won't be consistently
        # large enough to overcome the t*s2 growth if s2 > 0 and S(n) > x.
        # If s2 = 0, S(t) = s1 - dp[t]. dp[t] - dp[t-1] >= 0. S(t) is non-increasing.
        # If S(n) > x, S(t) >= S(n) > x for t < n. So if S(n) > x, impossible.
        # Thus, checking up to t=n is sufficient.
        for t in range(1, n + 1):
            current_sum = s1 + t * s2 - dp[t]
            if current_sum <= x:
                return t
        
        # If the loop finishes without finding a time t <= n that satisfies the condition,
        # based on the reasoning above, it's impossible.
        return -1