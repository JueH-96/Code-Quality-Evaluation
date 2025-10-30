class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        uniqueness_array = []
        for i in range(n):
            for j in range(i, n):
                uniqueness_array.append(len(set(nums[i:j+1])))
        uniqueness_array.sort()
        return uniqueness_array[(len(uniqueness_array) - 1) // 2]