class Solution:
    def minimumCost(self, nums: list[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        nums.sort()
        n = len(nums)
        median = nums[n // 2]
        
        # Find the closest palindromic number to the median
        palindromic_number = median
        while not is_palindrome(palindromic_number):
            palindromic_number -= 1
        
        # Calculate the cost to make all elements equal to the palindromic number
        cost = sum(abs(num - palindromic_number) for num in nums)
        
        return cost