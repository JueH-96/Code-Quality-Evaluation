class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        for i in range(len(nums)):
            if nums[i] > 0:
                if nums[i] < k:
                    return False
                else:
                    nums[i:i+k] = [n-1 for n in nums[i:i+k]]
        return True