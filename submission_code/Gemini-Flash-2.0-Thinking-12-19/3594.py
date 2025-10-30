class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        max_outlier = -float('inf')
        for sum_index in range(n):
            for outlier_index in range(n):
                if sum_index == outlier_index:
                    continue
                special_numbers = []
                for i in range(n):
                    if i != sum_index and i != outlier_index:
                        special_numbers.append(nums[i])
                if len(special_numbers) != n - 2:
                    continue
                sum_special = sum(special_numbers)
                if sum_special == nums[sum_index]:
                    max_outlier = max(max_outlier, nums[outlier_index])
        return max_outlier