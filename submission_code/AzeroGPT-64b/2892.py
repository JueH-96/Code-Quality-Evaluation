class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = len(nums) - 1
        c = Counter(nums)

        for i in range(1, n):
            if c[i] != 1:
                return False
        
        return c[n] == 2