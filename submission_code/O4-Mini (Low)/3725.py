from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Helper to compute sum of contributions using boundaries L and R and length limit k
        def count_subarrays(L: int, R: int, k: int) -> int:
            # We count pairs (a, b) with 0 <= a <= L, 0 <= b <= R, a + b + 1 <= k
            # Let A = min(L, k-1). Only a up to A contribute.
            A = min(L, k - 1)
            if A < 0:
                return 0
            # If R >= k-1, then for all a in [0..A]:
            #   b_max = min(R, k-1-a) = k-1-a
            #   so count = sum_{a=0..A} (k - a)
            if R >= k - 1:
                # sum_{a=0..A} (k - a) = (A+1)*k - (A*(A+1))//2
                return (A + 1) * k - (A * (A + 1)) // 2
            # Otherwise R < k-1. Let M be first a where k-1-a <= R:
            #   Solve k-1-a <= R  => a >= k-1-R
            M = max(0, k - 1 - R)
            # a in [0..M-1]: b_max = R => contributes (R+1) each
            cnt1 = (R + 1) * M
            # a in [M..A]: b_max = k-1-a => contributes (k - a) each
            if A >= M:
                length = A - M + 1
                # sum_{a=M..A} (k - a) = length*k - sum_{a=M..A} a
                sum_a = (A * (A + 1) // 2) - ((M - 1) * M // 2)
                cnt2 = length * k - sum_a
            else:
                cnt2 = 0
            return cnt1 + cnt2

        # Monotonic stack to get prev and next smaller (or larger) indices
        def boundaries(is_min: bool):
            prev = [-1] * n
            next_ = [n] * n
            stack = []
            for i in range(n):
                # For min: we want prev strictly less, next less-or-equal
                # For max: prev strictly greater, next greater-or-equal
                while stack:
                    j = stack[-1]
                    if (is_min and nums[j] > nums[i]) or (not is_min and nums[j] < nums[i]):
                        # j cannot dominate i for prev boundary
                        stack.pop()
                    else:
                        break
                prev[i] = stack[-1] if stack else -1
                stack.append(i)
            # clear and do next
            stack.clear()
            for i in range(n-1, -1, -1):
                while stack:
                    j = stack[-1]
                    if (is_min and nums[j] >= nums[i]) or (not is_min and nums[j] <= nums[i]):
                        stack.pop()
                    else:
                        break
                next_[i] = stack[-1] if stack else n
                stack.append(i)
            return prev, next_

        total = 0
        # sum of minimums
        prev_min, next_min = boundaries(is_min=True)
        for i in range(n):
            L = i - prev_min[i] - 1
            R = next_min[i] - i - 1
            cnt = count_subarrays(L, R, k)
            total += nums[i] * cnt

        # sum of maximums
        prev_max, next_max = boundaries(is_min=False)
        for i in range(n):
            L = i - prev_max[i] - 1
            R = next_max[i] - i - 1
            cnt = count_subarrays(L, R, k)
            total += nums[i] * cnt

        return total