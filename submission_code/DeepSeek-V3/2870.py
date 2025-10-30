class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i+1] - nums[i] == 1:
                current_len = 2
                for j in range(i+2, n):
                    if (j - i) % 2 == 0:
                        if nums[j] != nums[i]:
                            break
                    else:
                        if nums[j] != nums[i+1]:
                            break
                    current_len += 1
                if current_len > max_len:
                    max_len = current_len
        return max_len if max_len >= 2 else -1