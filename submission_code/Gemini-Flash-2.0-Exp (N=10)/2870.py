class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                sub_len = j - i + 1
                if sub_len > 1:
                    is_alternating = True
                    for k in range(1, sub_len):
                        if (nums[i+k] - nums[i+k-1]) != (1 if k % 2 == 1 else -1):
                            is_alternating = False
                            break
                    if is_alternating:
                        max_len = max(max_len, sub_len)
        return max_len