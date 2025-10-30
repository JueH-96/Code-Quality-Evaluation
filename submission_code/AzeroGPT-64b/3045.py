class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        inversions = 0
        total = 0
        
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                inversions += 1
                total += i + 1

        if inversions > 1 or (inversions == 1 and nums[-1] > nums[0]):
            return -1

        return (len(nums) - 1) - (total // inversions) if inversions else 0