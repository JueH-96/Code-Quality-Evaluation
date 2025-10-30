class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness_counts = []
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_count = len(set(subarray))
                uniqueness_counts.append(distinct_count)
        uniqueness_counts.sort()
        length = len(uniqueness_counts)
        if length % 2 == 1:
            median_index = length // 2
            return uniqueness_counts[median_index]
        else:
            median_index1 = length // 2 - 1
            median_index2 = length // 2
            return uniqueness_counts[median_index1]