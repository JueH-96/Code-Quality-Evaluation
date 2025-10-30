class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]

        def is_good(num):
            digits = [int(d) for d in str(num)]
            for i in range(len(digits)):
                if digits[i] != digits[-i-1]:
                    return False
            return True

        count = 0
        lower_bound = 10**(n-1)
        upper_bound = 10**n
        for num in range(lower_bound, upper_bound):
            if is_palindrome(num) and is_good(num):
                count += 1
        return count