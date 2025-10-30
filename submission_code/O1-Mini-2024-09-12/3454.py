class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        total_operations = 0
        prev_diff_inc = 0
        prev_diff_dec = 0

        for num, tar in zip(nums, target):
            current_diff = tar - num
            if current_diff > 0:
                total_operations += max(current_diff - prev_diff_inc, 0)
                prev_diff_inc = current_diff
                prev_diff_dec = 0
            elif current_diff < 0:
                current_diff = -current_diff
                total_operations += max(current_diff - prev_diff_dec, 0)
                prev_diff_dec = current_diff
                prev_diff_inc = 0
            else:
                prev_diff_inc = 0
                prev_diff_dec = 0

        return total_operations