class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        while nums:
            arr.append(nums.pop(0))
            if nums:
                arr.append(nums.pop())
        return arr