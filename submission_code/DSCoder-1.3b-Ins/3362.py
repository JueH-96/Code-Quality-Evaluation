class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness = sorted(set(nums))
        n = len(uniqueness)
        if n % 2 == 0:
            return (uniqueness[n // 2 - 1] + uniqueness[n // 2]) / 2
        else:
            return uniqueness[n // 2]