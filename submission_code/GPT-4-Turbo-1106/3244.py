class Solution:
    def minimumArrayLength(self, nums: list[int]) -> int:
        # The minimum length will be the count of non-zero elements in the gcd of all numbers
        from math import gcd
        from functools import reduce
        
        # Find the gcd of all elements in the array
        all_gcd = reduce(gcd, nums)
        
        # If the gcd is greater than 1, we can reduce the array to a single element which is the gcd itself
        if all_gcd > 1:
            return 1
        
        # If the gcd is 1, we can't reduce the array further than two elements
        return 2 if any(num > 1 for num in nums) else 1

# Example usage:
# sol = Solution()
# print(sol.minimumArrayLength([1,4,3,1]))  # Output: 1
# print(sol.minimumArrayLength([5,5,5,10,5]))  # Output: 2
# print(sol.minimumArrayLength([2,3,4]))  # Output: 1