class Solution:
    def minElement(self, nums: List[int]) -> int:
        
        def sum_digits(n):
            s = 0
            while n:
                s += n % 10
                n //= 10
            return s

        for i in range(len(nums)):
            nums[i] = sum_digits(nums[i])

        return min(nums)