class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness_array = []
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                distinct_count = len(set(subarray))
                uniqueness_array.append(distinct_count)
        
        uniqueness_array.sort()
        
        n = len(uniqueness_array)
        if n % 2 == 0:
            median = uniqueness_array[n // 2 - 1]
        else:
            median = uniqueness_array[n // 2]
        
        return median