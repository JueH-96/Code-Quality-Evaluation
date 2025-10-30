from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        # Let d[i] = target[i] - nums[i]
        # We claim that the minimum number of operations is
        #     max(total_positive_change, total_negative_change)
        #
        # Explanation:
        # Any operation on a contiguous subarray (adding or subtracting 1)
        # can be seen in a “difference” view. Define:
        #   b[0] = d[0], and for i>=1, b[i] = d[i] - d[i-1].
        # When you add 1 on an interval [l, r], you add +1 to b[l] and subtract 1
        # from b[r+1] (if r+1 < n). Similarly for subtracting 1.
        # Thus, to “build” d we need to supply b by a sum of such operations.
        #
        # Notice that every operation increments one “+1” somewhere and a corresponding
        # “-1” somewhere else (if within bounds). Therefore, the minimal number
        # operations required must be enough to supply |b[i]| amount of change.
        # More precisely, if we let:
        #   pos = max(0, d[0]) + sum( max(0, d[i] - d[i-1]) for i=1..n-1 )
        #   neg = max(0, -d[0]) + sum( max(0, d[i-1] - d[i]) for i=1..n-1 )
        # then the answer is max(pos, neg).
        #
        # To see this in our examples:
        # Example 1:
        #   nums   = [3, 5, 1, 2]
        #   target = [4, 6, 2, 4]
        #   d      = [1, 1, 1, 2]
        #   pos = max(0, 1) + max(0, 1-1) + max(0, 1-1) + max(0, 2-1) = 1 + 0 + 0 + 1 = 2
        #   neg = max(0, -1) + ... = 0, so answer = max(2, 0) = 2.
        #
        # Example 2:
        #   nums   = [1, 3, 2]
        #   target = [2, 1, 4]
        #   d      = [1, -2, 2]
        #   pos = max(0, 1) + max(0, -2-1) + max(0, 2-(-2)) = 1 + 0 + 4 = 5
        #   neg = max(0, -1) + max(0, 1-(-2)) + max(0, -2-2) = 0 + 3 + 0 = 3
        #   answer = max(5, 3) = 5
        #
        # This method works in O(n) and meets the constraints.
        
        # Compute initial difference d[0]
        first_diff = target[0] - nums[0]
        pos = max(0, first_diff)
        neg = max(0, -first_diff)
        
        # For i>=1, let diff_change = (target[i]-nums[i]) - (target[i-1]-nums[i-1])
        for i in range(1, n):
            current_diff = target[i] - nums[i]
            previous_diff = target[i-1] - nums[i-1]
            delta = current_diff - previous_diff
            if delta > 0:
                pos += delta
            else:
                neg += -delta  # same as max(0, -delta)
                
        return max(pos, neg)