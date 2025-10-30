class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        total = 0
        prev = 0
        for i in range(n):
            if i == 0 or nums[i] != nums[i-1]:
                current = prev + 1
            else:
                current = 1
            total += current
            prev = current
        return total