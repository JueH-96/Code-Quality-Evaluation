class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        n = len(nums)
        arr = []
        while n > 1:
            if n % 2 == 0:
                arr.append(min(nums))
                nums.remove(min(nums))
            else:
                arr.append(max(nums))
                nums.remove(max(nums))
            n -= 1
        return arr