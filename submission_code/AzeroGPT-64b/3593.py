class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def gcd(a, b) -> int:
            return a if b == 0 else gcd(b, a % b)

        def lcm(a, b) -> int:
            return a * b // gcd(a, b)

        best = 0
        gc = nums[0]
        lc = nums[0]
        for i in range(1, len(nums)):
            gc = gcd(gc, nums[i])
            lc = lcm(lc, nums[i])
            if i == len(nums) - 1:
                best = gc * lc
            best = max(best, gc * nums[i], gcd(gc * nums[i], nums[i]) * lcm(lc, nums[i]))
        return best