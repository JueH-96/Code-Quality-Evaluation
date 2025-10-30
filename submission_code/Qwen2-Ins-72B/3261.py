class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for bit in range(30, -1, -1):
            zero, one = 0, 0
            for num in nums:
                if num & (1 << bit):
                    one += 1
                else:
                    zero += 1
            if one == 0:
                continue
            if zero <= k:
                ans |= (1 << bit)
            else:
                break
        return ans