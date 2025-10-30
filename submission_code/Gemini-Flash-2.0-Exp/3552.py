class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            largest_digit = 9
            while largest_digit > 0:
                if largest_digit % k == 0:
                    return str(largest_digit)
                largest_digit -= 1
            return ""
        
        max_num = 10**n - 1
        min_num = 10**(n-1)
        
        for num in range(max_num, min_num - 1, -1):
            num_str = str(num)
            if num_str == num_str[::-1] and num % k == 0:
                return num_str
        
        return ""