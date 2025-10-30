class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = [0] * 3
        for num in nums:
            count[num % 3] += 1
        return min(count[0] + count[2], count[1] + (count[0] if count[2] % 2 == 1 else count[2] - 1) + 1)