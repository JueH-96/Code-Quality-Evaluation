class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = 0
        for num in nums:
            rem = num % 3
            if rem == 1:
                count += 1
            elif rem == 2:
                count += 1
        return count