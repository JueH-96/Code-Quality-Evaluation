from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i + 1] != nums[i] + 1:
                continue
            current_len = 2
            if current_len > max_len:
                max_len = current_len
            expected_even = nums[i]
            expected_odd = nums[i + 1]
            for j in range(i + 2, n):
                step = j - i
                if step % 2 == 0:
                    expected = expected_even
                else:
                    expected = expected_odd
                if nums[j] == expected:
                    current_len += 1
                    if current_len > max_len:
                        max_len = current_len
                else:
                    break
        return max_len if max_len >= 2 else -1