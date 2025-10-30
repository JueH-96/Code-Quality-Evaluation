class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(s):
            return s == s[::-1]

        def can_form_k_palindromic(s, k):
            from collections import Counter
            count = Counter(s)
            half = n // 2
            for digit in count:
                if count[digit] % 2 != 0:
                    half -= 1
                half -= count[digit] // 2
                if half < 0:
                    return False
            return True

        def generate_palindromes(n):
            if n == 1:
                return [str(i) for i in range(1, 10)]
            half = n // 2
            palindromes = []
            for i in range(10**(half-1), 10**half):
                s = str(i)
                if n % 2 == 0:
                    palindromes.append(s + s[::-1])
                else:
                    for j in range(10):
                        palindromes.append(s + str(j) + s[::-1])
            return palindromes

        palindromes = generate_palindromes(n)
        good_count = 0
        for palindrome in palindromes:
            if int(palindrome) % k == 0:
                good_count += 1
            elif can_form_k_palindromic(palindrome, k):
                good_count += 1

        return good_count