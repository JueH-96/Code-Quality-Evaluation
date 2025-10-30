import collections
import itertools

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # P[k] stores the prefix sum of nums[0...k-1]. P[0] = 0.
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]
            
        # dp[i] stores the maximum length of a non-decreasing array
        # that can be formed using the prefix nums[0...i-1].
        dp = [0] * (n + 1)
        
        # last_val[i] stores the minimum possible sum of the last segment
        # when achieving dp[i] length for nums[0...i-1].
        # last_val[0] is initialized to 0, which conceptually allows any positive first segment.
        last_val = [0] * (n + 1) 
        
        # ptr is the "two-pointer" for finding the optimal previous split point.
        # This pointer `ptr` ensures that `last_val[ptr]` is always less than or equal to `P[i] - P[ptr]`.
        # It's an index into the `dp` and `last_val` arrays.
        ptr = 0

        for i in range(1, n + 1):
            # This loop ensures that `ptr` points to the first index `j`
            # such that `P[i] - P[j] >= last_val[j]`.
            # If `P[i] - P[ptr]` is too small to be a valid sum for `last_val[ptr]`,
            # we must advance `ptr`.
            # We continue advancing `ptr` as long as the condition `P[i] - P[ptr] < last_val[ptr]` holds.
            # This is equivalent to `P[i] < P[ptr] + last_val[ptr]`.
            # `P[ptr] + last_val[ptr]` is implicitly maintained in a non-decreasing manner by the problem structure.
            while ptr < i and P[i] - P[ptr] < last_val[ptr]:
                ptr += 1
            
            # Now, `ptr` points to the optimal previous split point (j) for `i`.
            # This is because the sequence of sums S_1, S_2, ... S_k must be non-decreasing.
            # `dp[ptr]` is the maximum length achieved for `nums[0...ptr-1]`.
            # `last_val[ptr]` is the minimum last sum for `nums[0...ptr-1]`.
            # By choosing `nums[ptr...i-1]` as the new segment:
            # The new length is `dp[ptr] + 1`.
            # The new last sum is `P[i] - P[ptr]`.
            dp[i] = dp[ptr] + 1
            last_val[i] = P[i] - P[ptr]

        return dp[n]