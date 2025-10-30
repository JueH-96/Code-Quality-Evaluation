class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        diff = [0]*len(nums)
        acc = 0
        for i in range(len(nums)):
            if i >= k:
                acc -= diff[i-k]
            acc += nums[i]
            diff[i] = acc
            if i >= k-1 and acc != 0:
                return False
        return True