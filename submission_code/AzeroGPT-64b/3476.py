class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = 0
        for n in nums:
            count += n % 3
        if count > len(nums) // 3 * 3:
            return len(nums) - count
        else:
            return count