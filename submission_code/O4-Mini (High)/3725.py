from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Previous Less Element for minimums (strictly less)
        pleMin = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            pleMin[i] = stack[-1] if stack else -1
            stack.append(i)
        # Next Less-or-Equal Element for minimums
        nleMin = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                j = stack.pop()
                nleMin[j] = i
            stack.append(i)
        # Previous Greater Element for maximums (strictly greater)
        pleMax = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            pleMax[i] = stack[-1] if stack else -1
            stack.append(i)
        # Next Greater-or-Equal Element for maximums
        nleMax = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                j = stack.pop()
                nleMax[j] = i
            stack.append(i)

        ans = 0
        # Loop over each position as the candidate min or max
        for i in range(n):
            v = nums[i]
            # the earliest allowed start index for length ≤ k
            ll = i - k + 1

            # ---- Minimum contributions ----
            llow = pleMin[i] + 1
            lhigh = i
            l0 = max(llow, ll)
            count_min = 0
            if l0 <= lhigh:
                # break point where l + k - 1 <= (nleMin[i] - 1)
                x = nleMin[i] - k
                # first batch: l from l0 to y1 = min(lhigh, x)
                y1 = min(lhigh, x)
                if y1 >= l0:
                    t1 = y1 - l0 + 1
                    # sum of l over [l0..y1]
                    sum_l = (l0 + y1) * t1 // 2
                    # sum of (l + k - i)
                    count_min = sum_l + t1 * (k - i)
                # second batch: l from x+1 to lhigh
                x2 = max(l0, x + 1)
                if x2 <= lhigh:
                    t2 = lhigh - x2 + 1
                    # each such l gives right_count = nleMin[i] - i
                    count_min += t2 * (nleMin[i] - i)

            # ---- Maximum contributions ----
            llow = pleMax[i] + 1
            lhigh = i
            l0 = max(llow, ll)
            count_max = 0
            if l0 <= lhigh:
                x = nleMax[i] - k
                y1 = min(lhigh, x)
                if y1 >= l0:
                    t1 = y1 - l0 + 1
                    sum_l = (l0 + y1) * t1 // 2
                    count_max = sum_l + t1 * (k - i)
                x2 = max(l0, x + 1)
                if x2 <= lhigh:
                    t2 = lhigh - x2 + 1
                    count_max += t2 * (nleMax[i] - i)

            ans += v * (count_min + count_max)

        return ans