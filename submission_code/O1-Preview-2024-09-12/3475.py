class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        K = 3
        flip = 0
        res = 0
        flip_end = [0] * n
        
        for i in range(n):
            if i >= K:
                flip -= flip_end[i - K]
            if nums[i] ^ (flip % 2) == 0:
                if i + K > n:
                    return -1
                flip += 1
                flip_end[i] = 1
                res += 1
        return res