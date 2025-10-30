class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        sum_total = sum(nums)
        elements = set(nums)
        possible_outliers = []
        for x in nums:
            outlier = sum_total - 2 * x
            if outlier in elements:
                possible_outliers.append(outlier)
        return max(possible_outliers)