from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        T = k + 1                    # we will need a+b ≤ T   ( because length = a+b-1 ≤ k )

        # ---------- helpers ----------
        def pairs_not_longer_than(A: int, B: int) -> int:
            """
            number of pairs (a,b)  with 1≤a≤A , 1≤b≤B and a+b ≤ T   (T is captured from outer scope)
            derived as explained in analysis – runs in O(1)
            """
            # maximal possible a (cannot exceed A and also must have at least b=1 left)
            U = min(A, T - 1)
            if U <= 0:
                return 0

            # if the smallest possible (T-U) is already greater than B,
            # every allowed a combines with all B possibilities
            if (T - U) > B:
                return U * B

            # otherwise we have two zones:
            #    1) a < a_start  -> all B values work
            #    2) a ≥ a_start  -> b is limited to (T-a)
            a_start = max(1, T - B)          # first a where (T-a) ≤ B
            first_part  = (a_start - 1) * B  # zone-1 contribution

            m = U - a_start + 1              # amount of a’s in zone-2
            # sum_{a=a_start}^{U} (T-a)  == m*T − sum_{a=a_start}^{U} a
            sum_a = (a_start + U) * m // 2
            second_part = m * T - sum_a      # zone-2 contribution

            return first_part + second_part
        # --------------------------------

        # previous less (strict) for minima
        prev_less = [-1] * n
        stack = []
        for i, v in enumerate(nums):
            while stack and nums[stack[-1]] >= v:
                stack.pop()
            prev_less[i] = stack[-1] if stack else -1
            stack.append(i)

        # next less-or-equal for minima
        next_le   = [n] * n
        stack.clear()
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            next_le[i] = stack[-1] if stack else n
            stack.append(i)

        # previous greater (strict) for maxima
        prev_greater = [-1] * n
        stack.clear()
        for i, v in enumerate(nums):
            while stack and nums[stack[-1]] <= v:
                stack.pop()
            prev_greater[i] = stack[-1] if stack else -1
            stack.append(i)

        # next greater-or-equal for maxima
        next_ge = [n] * n
        stack.clear()
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            next_ge[i] = stack[-1] if stack else n
            stack.append(i)

        # accumulate answer
        ans = 0
        for i in range(n):
            # minima contribution
            left  = i - prev_less[i]
            right = next_le[i] - i
            cnt_min = pairs_not_longer_than(left, right)

            # maxima contribution
            left  = i - prev_greater[i]
            right = next_ge[i] - i
            cnt_max = pairs_not_longer_than(left, right)

            ans += nums[i] * (cnt_min + cnt_max)

        return ans