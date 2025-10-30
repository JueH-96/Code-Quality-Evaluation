class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness_array = []
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_count = len(set(subarray))
                uniqueness_array.append(distinct_count)
        uniqueness_array.sort()
        array_len = len(uniqueness_array)
        if array_len % 2 == 1:
            median_index = array_len // 2
        else:
            median_index = array_len // 2 - 1
        return uniqueness_array[median_index]