class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        expected_count = {}
        for i in range(1, n):
            expected_count[i] = 1
        expected_count[n] = 2

        return count == expected_count