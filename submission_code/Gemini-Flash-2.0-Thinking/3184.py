from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        b = [nums[i] - i for i in range(n)]

        unique_b = sorted(list(set(b)))
        b_map = {val: i for i, val in enumerate(unique_b)}

        m = len(unique_b)
        fenwick = [-float('inf')] * (m + 1)

        def update(index, val):
            index += 1
            while index <= m:
                fenwick[index] = max(fenwick[index], val)
                index += index & (-index)

        def query(index):
            index += 1
            max_val = -float('inf')
            while index > 0:
                max_val = max(max_val, fenwick[index])
                index -= index & (-index)
            return max_val

        max_balanced_sum = -float('inf')

        for i in range(n):
            current_b = b[i]
            current_num = nums[i]

            if current_b in b_map:
                idx = b_map[current_b]
                max_prev_sum = query(idx)

                current_sum = current_num
                if max_prev_sum > -float('inf'):
                    current_sum += max_prev_sum

                max_balanced_sum = max(max_balanced_sum, current_sum)
                update(idx, current_sum)
            else:
                max_balanced_sum = max(max_balanced_sum, current_num)
                # This case should not happen if b_map is correctly constructed

        # Consider subsequences of length 1
        max_single_element = max(nums) if nums else -float('inf')
        return max(max_balanced_sum, max_single_element)