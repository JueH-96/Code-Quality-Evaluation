class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        nums.sort()
        l, r = 0, len(nums) - 1
        while l <= r:
            arr.append(nums[r])
            r -= 1
            if l <= r:
                arr.append(nums[l])
                l += 1

        return arr