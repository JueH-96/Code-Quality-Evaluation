class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness = []
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                uniqueness.append(len(set(nums[i:j+1])))
        uniqueness.sort()
        return uniqueness[len(uniqueness)//2]