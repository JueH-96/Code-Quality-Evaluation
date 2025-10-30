class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1
        for i in range(n - 1):
            if nums[i + 1] - nums[i] != 1:
                continue
            current_length = 2
            if current_length > max_len:
                max_len = current_length
            sign = -1
            j = 1
            while i + j + 1 < n:
                if nums[i + j + 1] - nums[i + j] == sign:
                    current_length = j + 2
                    if current_length > max_len:
                        max_len = current_length
                    sign = -sign
                    j += 1
                else:
                    break
        return max_len