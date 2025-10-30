class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            arr.append(min(nums))
            nums.remove(min(nums))
            arr.append(min(nums))
            nums.remove(min(nums))
        return arr