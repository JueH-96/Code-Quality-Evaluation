import collections
from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # pg[i]: index of the first element to the left of i strictly greater than nums[i]
        pg = [-1] * n
        stack = [] # stores indices i in increasing order, nums[i] in decreasing order
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                pg[i] = stack[-1]
            stack.append(i)

        # ng[i]: index of the first element to the right of i greater than or equal to nums[i]
        ng = [n] * n
        stack = [] # stores indices i in decreasing order, nums[i] in non-increasing order
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]: # Note: < for ng to give >=
                stack.pop()
            if stack:
                ng[i] = stack[-1]
            stack.append(i)

        # ps[i]: index of the first element to the left of i strictly smaller than nums[i]
        ps = [-1] * n
        stack = [] # stores indices i in increasing order, nums[i] in increasing order
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                ps[i] = stack[-1]
            stack.append(i)

        # ns[i]: index of the first element to the right of i smaller than or equal to nums[i]
        ns = [n] * n
        stack = [] # stores indices i in decreasing order, nums[i] in non-decreasing order
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]: # Note: > for ns to give <=
                stack.pop()
            if stack:
                ns[i] = stack[-1]
            stack.append(i)

        # Calculate total sum of maximums and minimums
        sum_max = 0
        sum_min = 0

        for i in range(n):
            # Contribution of nums[i] as maximum
            # nums[i] is max in subarrays [l, r] where pg[i] < l <= i <= r < ng[i] and r - l + 1 <= k
            L_max = pg[i] + 1 # Smallest possible start index l > pg[i]
            R_max = ng[i] - 1 # Largest possible end index r < ng[i]

            # Sum over possible left endpoints l for this i (where i is the max)
            # l ranges from L_max to i
            # For a fixed l, valid range for r is [i, min(R_max, l + k - 1)]

            # Split the summation over l in [L_max, i]
            # Part 1: l such that min(R_max, l + k - 1) = l + k - 1 => l + k - 1 <= R_max => l <= R_max - k + 1
            # l is in [L_max, min(i, R_max - k + 1)]
            l_end1 = min(i, R_max - k + 1)
            if L_max <= l_end1:
                # For each l in [L_max, l_end1], number of r is (l + k - 1) - i + 1 = l + k - i
                # Sum of (l + k - i) for l from L_max to l_end1
                # Arithmetic progression: sum(l + C) where C = k - i
                count_l = l_end1 - L_max + 1
                C = k - i
                sum_of_l_in_range = (L_max + l_end1) * count_l // 2
                sum_of_counts_r = sum_of_l_in_range + C * count_l
                sum_max += nums[i] * sum_of_counts_r

            # Part 2: l such that min(R_max, l + k - 1) = R_max => l + k - 1 > R_max => l > R_max - k + 1
            # l is in [max(L_max, R_max - k + 2), i]
            l_start2 = max(L_max, R_max - k + 2)
            if l_start2 <= i:
                 # For each l in [l_start2, i], number of r is R_max - i + 1 (constant)
                count_l = i - l_start2 + 1
                count_r = R_max - i + 1
                sum_of_counts_r = count_r * count_l
                sum_max += nums[i] * sum_of_counts_r

            # Contribution of nums[i] as minimum
            # nums[i] is min in subarrays [l, r] where ps[i] < l <= i <= r < ns[i] and r - l + 1 <= k
            L_min = ps[i] + 1 # Smallest possible start index l > ps[i]
            R_min = ns[i] - 1 # Largest possible end index r < ns[i]

            # Sum over possible left endpoints l for this i (where i is the min)
            # l ranges from L_min to i
            # For a fixed l, valid range for r is [i, min(R_min, l + k - 1)]

            # Split the summation over l in [L_min, i]
            # Part 1: l such that min(R_min, l + k - 1) = l + k - 1 => l <= R_min - k + 1
            # l is in [L_min, min(i, R_min - k + 1)]
            l_end1 = min(i, R_min - k + 1)
            if L_min <= l_end1:
                # For each l in [L_min, l_end1], number of r is (l + k - 1) - i + 1 = l + k - i
                # Sum of (l + k - i) for l from L_min to l_end1
                count_l = l_end1 - L_min + 1
                C = k - i
                sum_of_l_in_range = (L_min + l_end1) * count_l // 2
                sum_of_counts_r = sum_of_l_in_range + C * count_l
                sum_min += nums[i] * sum_of_counts_r

            # Part 2: l such that min(R_min, l + k - 1) = R_min => l > R_min - k + 1
            # l is in [max(L_min, R_min - k + 2), i]
            l_start2 = max(L_min, R_min - k + 2)
            if l_start2 <= i:
                 # For each l in [l_start2, i], number of r is R_min - i + 1 (constant)
                count_l = i - l_start2 + 1
                count_r = R_min - i + 1
                sum_of_counts_r = count_r * count_l
                sum_min += nums[i] * sum_of_counts_r

        total_sum = sum_max + sum_min
        return total_sum