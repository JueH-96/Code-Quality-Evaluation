class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            rem = num % 3
            if rem == 1:
                ans += 1
            elif rem == 2:
                ans += 1
        return ans