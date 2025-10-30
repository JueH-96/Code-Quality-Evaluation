class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n):
            if i + 1 >= n:
                break
            if nums[i + 1] != nums[i] + 1:
                continue
            # Found a valid starting point
            current_len = 2
            max_len = max(max_len, current_len)
            expected_diff = -1
            j = i + 2
            while j < n:
                actual_diff = nums[j] - nums[j - 1]
                if actual_diff != expected_diff:
                    break
                current_len += 1
                max_len = max(max_len, current_len)
                expected_diff *= -1
                j += 1
        return max_len if max_len != -1 else -1