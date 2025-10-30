class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]

        # Start from the largest n-digit number
        max_num = int('9' * n)
        
        # Iterate downwards to find the largest k-palindromic number
        for num in range(max_num, 0, -1):
            if num % k == 0 and is_palindrome(num):
                return str(num)
        
        return ""

# Example usage:
# sol = Solution()
# print(sol.largestPalindrome(3, 5))  # Output: "595"
# print(sol.largestPalindrome(1, 4))  # Output: "8"
# print(sol.largestPalindrome(5, 6))  # Output: "89898"