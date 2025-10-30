class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        uniqueness_array = []
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                uniqueness_array.append(len(set(subarray)))
        
        uniqueness_array.sort()
        
        median_index = len(uniqueness_array) // 2
        
        return uniqueness_array[median_index]