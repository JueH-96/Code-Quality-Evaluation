class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        for num in nums:
            r = num % 3
            # Cost to make num % 3 == 0 is the minimum of r and 3-r
            operations += min(r, 3 - r)
        return operations