class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindromic(n):
            return str(n) == str(n)[::-1]
        nums.sort()
        median = nums[len(nums)//2]
        nearest_palindrome = -1
        def closest_palindrome(median):
            smaller = median
            larger = median
            while True:
                if is_palindromic(smaller):
                    return smaller
                if is_palindromic(larger):
                    return larger
                smaller -= 1
                larger += 1
        nearest_palindrome = closest_palindrome(median)
        cost = sum([abs(num - nearest_palindrome) for num in nums])
        return cost