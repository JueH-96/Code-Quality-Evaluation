class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness_array = []
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                uniqueness_array.append(len(set(sub_array)))
        uniqueness_array.sort()
        
        length = len(uniqueness_array)
        if length % 2 == 0:
            median = uniqueness_array[length // 2 -1]
        else:
            median = uniqueness_array[length // 2]
        return median