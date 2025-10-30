class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1
        for i in range(n - 1):
            if nums[i+1] - nums[i] == 1:
                current_len = 2
                for j in range(i + 2, n):
                    if nums[j] - nums[j-1] == (1 if (j - i) % 2 == 1 else -1):
                        current_len += 1
                    else:
                        break
                max_len = max(max_len, current_len)
        return max_len