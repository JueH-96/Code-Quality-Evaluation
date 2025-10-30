class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            for j in range(i + 1, n):
                length = j - i + 1
                if length > 1:
                    is_alternating = True
                    if nums[i + 1] - nums[i] != 1:
                        is_alternating = False
                    else:
                        for k in range(i + 2, j + 1):
                            if nums[k] - nums[k - 1] != (-1)**(k - i):
                                is_alternating = False
                                break
                    if is_alternating:
                        max_len = max(max_len, length)
        return max_len