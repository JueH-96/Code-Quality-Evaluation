class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        max_outlier = -float('inf')

        for i in range(n):
            potential_outlier = nums[i]
            remaining_nums = nums[:i] + nums[i+1:]
            remaining_nums.sort()

            n_minus_2 = n - 2

            # Case 1: Smallest n-2 numbers are special
            if n_minus_2 > 0:
                special_numbers_small = remaining_nums[:n_minus_2]
                sum_special_small = sum(special_numbers_small)
                if sum_special_small in remaining_nums:
                    temp_remaining = list(remaining_nums)
                    for val in special_numbers_small:
                        temp_remaining.remove(val)
                    if sum_special_small in temp_remaining:
                        max_outlier = max(max_outlier, potential_outlier)

            # Case 2: Largest n-2 numbers are special
            if n_minus_2 > 0:
                special_numbers_large = remaining_nums[len(remaining_nums) - n_minus_2:]
                sum_special_large = sum(special_numbers_large)
                if sum_special_large in remaining_nums:
                    temp_remaining = list(remaining_nums)
                    for val in special_numbers_large:
                        temp_remaining.remove(val)
                    if sum_special_large in temp_remaining:
                        max_outlier = max(max_outlier, potential_outlier)

        return max_outlier