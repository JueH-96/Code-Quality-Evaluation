class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                sub_array = nums[i:j+1]
                m = len(sub_array)
                if m > 1:
                    is_alternating = True
                    if sub_array[1] - sub_array[0] != 1:
                        is_alternating = False
                    else:
                        expected_diff = -1
                        for k in range(2, m):
                            if sub_array[k] - sub_array[k-1] != expected_diff:
                                is_alternating = False
                                break
                            expected_diff *= -1
                    if is_alternating:
                        max_len = max(max_len, m)
        return max_len