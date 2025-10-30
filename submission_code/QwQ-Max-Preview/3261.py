class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for j in reversed(range(30)):
            cnt = 0
            mask = 1 << j
            for num in nums:
                if (num & mask) == 0:
                    cnt += 1
            if cnt < (n - k):
                ans |= mask
        return ans