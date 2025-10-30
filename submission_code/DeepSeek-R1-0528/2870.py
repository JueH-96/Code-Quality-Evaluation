class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i + 1] - nums[i] == 1:
                current_len = 2
                max_len = max(max_len, current_len)
                step = -1
                j = i + 2
                while j < n:
                    if nums[j] == nums[j - 1] + step:
                        current_len += 1
                        max_len = max(max_len, current_len)
                        step *= -1
                        j += 1
                    else:
                        break
        return max_len