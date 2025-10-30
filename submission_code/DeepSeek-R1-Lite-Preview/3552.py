class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            # Single-digit palindromes are 1 to 9
            for x in range(9, 0, -1):
                if x % k == 0:
                    return str(x)
            return "1"  # Fallback, should not reach here

        m = (n + 1) // 2  # ceil(n/2)

        # Initialize A as the largest m-digit number
        A_digits = [9] * m

        def compute_P_mod_k(A_digits, n, k):
            m = len(A_digits)
            c = pow(10, n - m, k)
            reverse_A_mod_k = 0
            for digit in reversed(A_digits):
                reverse_A_mod_k = (reverse_A_mod_k * 10 + digit) % k
            A_mod_k = 0
            for digit in A_digits:
                A_mod_k = (A_mod_k * 10 + digit) % k
            P_mod_k = (A_mod_k * c + reverse_A_mod_k) % k
            return P_mod_k

        def decrement_A(A_digits):
            i = len(A_digits) - 1
            while i >= 0:
                if A_digits[i] > 0:
                    A_digits[i] -= 1
                    for j in range(i + 1, len(A_digits)):
                        A_digits[j] = 9
                    break
                else:
                    A_digits[i] = 9
                    i -= 1
            # Remove leading zero if any
            if A_digits[0] == 0:
                return False
            return True

        def construct_palindrome(A_digits, n):
            if n % 2 == 1:
                first_half = A_digits
                second_half = A_digits[:-1][::-1]
            else:
                first_half = A_digits
                second_half = A_digits[::-1]
            palindrome_digits = first_half + second_half
            return ''.join(str(digit) for digit in palindrome_digits)

        while True:
            P_mod_k = compute_P_mod_k(A_digits, n, k)
            if P_mod_k == 0:
                palindrome = construct_palindrome(A_digits, n)
                return palindrome
            if not decrement_A(A_digits):
                break

        return "No solution"  # As per constraints, this should not be reached