class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # helper function: construct palindrome from left half (in int form) string L.
        def make_palindrome(L: str) -> str:
            if n % 2 == 0:
                return L + L[::-1]
            else:
                # for odd, mirror all but the last digit.
                return L + L[:-1][::-1]
        
        # number of digits in the left half.
        m = (n + 1) // 2
        
        # Lower bound for left half (cannot have leading zero)
        lower_bound = 10 ** (m - 1)
        # Maximum left half "number" (all 9's) as an integer
        left = int("9" * m)
        
        # iterate downward from the maximum left half until we hit the lower_bound
        while left >= lower_bound:
            L_str = str(left)
            P = make_palindrome(L_str)
            # Check length safety (it must be exactly n digits)
            if len(P) != n:
                left -= 1
                continue
            # We can use int(P) % k since k is at most 9.
            if int(P) % k == 0:
                return P
            left -= 1
        
        # In a valid problem instance an answer always exists.
        return ""
        
# For testing with the provided examples
if __name__ == '__main__':
    sol = Solution()
    print(sol.largestPalindrome(3, 5))  # expected "595"
    print(sol.largestPalindrome(1, 4))  # expected "8"
    print(sol.largestPalindrome(5, 6))  # expected "89898"