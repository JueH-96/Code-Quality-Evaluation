class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            seen = [0] * (n + 1)
            cur = -1
            for j in range(i, n):
                x = nums[j]
                seen[x] = 1
                if j > i and (seen[x - 1] == 0 or seen[x + 1] == 0):
                    cur += 1
                res += cur
        return res