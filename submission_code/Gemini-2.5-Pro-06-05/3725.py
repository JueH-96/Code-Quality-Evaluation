from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of the maximum and minimum elements of all subarrays
        with at most k elements.
        """
        max_sum = self._calculate_contribution(nums, k, True)
        min_sum = self._calculate_contribution(nums, k, False)
        return max_sum + min_sum

    def _calculate_contribution(self, nums: List[int], k: int, is_max: bool) -> int:
        n = len(nums)
        
        # Find left boundaries for each element.
        # For max contribution, we need prev_greater (>).
        # For min contribution, we need prev_smaller (<).
        left = [-1] * n
        stack = []
        for i in range(n):
            # Pop from stack while the condition for the monotonic stack is violated.
            while stack and (nums[stack[-1]] <= nums[i] if is_max else nums[stack[-1]] >= nums[i]):
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)

        # Find right boundaries for each element.
        # For max: next_greater_equal (>=).
        # For min: next_smaller_equal (<=).
        right = [n] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and (nums[stack[-1]] < nums[i] if is_max else nums[stack[-1]] > nums[i]):
                stack.pop()
            if stack:
                right[i] = stack[-1]
            stack.append(i)

        total_contribution = 0
        for i in range(n):
            Li, Ri = left[i], right[i]
            
            # Count pairs (p, q) such that:
            # 1. Li < p <= i
            # 2. i <= q < Ri
            # 3. q - p + 1 <= k  => q <= p + k - 1
            
            # For a given p, number of valid q's is min(Ri-1, p+k-1) - i + 1.
            # We sum this over valid p's.
            # A p is valid if it's in (Li, i] and allows at least one q (p+k-1 >= i).
            
            p_start = max(Li + 1, i - k + 1)
            p_end = i

            if p_start > p_end:
                continue

            count = 0
            
            # The number of choices for q changes based on p.
            # The crossover point is p + k - 1 = Ri - 1, so p = Ri - k.
            p_split = Ri - k

            # Case 1: p <= p_split. Number of q's is (p + k - 1) - i + 1 = p + k - i
            # Sum of (p + k - i) for p from p_start to min(p_end, p_split)
            p_upper1 = min(p_end, p_split)
            if p_upper1 >= p_start:
                num_terms = p_upper1 - p_start + 1
                # Sum of arithmetic series for p
                sum_p = (p_start + p_upper1) * num_terms // 2
                # Sum of constant (k - i)
                sum_const = (k - i) * num_terms
                count += sum_p + sum_const
            
            # Case 2: p > p_split. Number of q's is (Ri - 1) - i + 1 = Ri - i
            # Sum of (Ri - i) for p from max(p_start, p_split + 1) to p_end
            p_lower2 = max(p_start, p_split + 1)
            if p_end >= p_lower2:
                num_terms = p_end - p_lower2 + 1
                sum_const = (Ri - i) * num_terms
                count += sum_const
            
            total_contribution += count * nums[i]
            
        return total_contribution