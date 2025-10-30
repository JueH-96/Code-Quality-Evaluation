class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]
        
        def find_nearest_palindrome(num):
            lower = num
            upper = num
            while True:
                if is_palindrome(lower):
                    return lower
                if is_palindrome(upper):
                    return upper
                lower -= 1
                upper += 1
        
        median = sorted(nums)[len(nums) // 2]
        nearest_palindrome = find_nearest_palindrome(median)
        
        return sum(abs(num - nearest_palindrome) for num in nums)