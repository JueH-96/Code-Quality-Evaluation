class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, -1, -1):
                if i % k == 0:
                    return str(i)
            return "0" # Should not reach here according to problem statement

        start_half_str = '9' * ((n + 1) // 2)
        start_half = int(start_half_str)
        end_half = 10 ** ((n + 1) // 2 - 1) if (n + 1) // 2 > 1 else 0

        for first_half_val in range(start_half, end_half - 1, -1):
            first_half = str(first_half_val)
            if n % 2 == 1:
                palindrome = first_half + first_half[:-1][::-1]
            else:
                palindrome = first_half + first_half[::-1]
            if int(palindrome) % k == 0:
                return palindrome
        return "0" # Should not reach here according to problem statement, but as a safe guard