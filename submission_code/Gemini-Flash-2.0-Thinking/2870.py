class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1

        for i in range(n - 1):
            for j in range(i + 1, n):
                subarray = nums[i : j + 1]
                m = len(subarray)

                if m > 1:
                    is_alternating = True
                    for k in range(1, m):
                        expected_diff = 1 if k % 2 == 1 else -1
                        if subarray[k] - subarray[k - 1] != expected_diff:
                            is_alternating = False
                            break
                    if is_alternating:
                        max_length = max(max_length, m)

        return max_length