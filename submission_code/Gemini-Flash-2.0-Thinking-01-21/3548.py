import collections

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        count = 0
        if n % 2 == 1:
            m = (n - 1) // 2
            for d1 in range(1, 10):
                for prefix_digits in self.generate_digit_sequences(m, 0):
                    for dm1 in range(0, 10):
                        first_half = [d1] + prefix_digits + [dm1]
                        palindrome_digits = first_half + list(reversed(prefix_digits)) + [d1]
                        palindrome_str = "".join(map(str, palindrome_digits))
                        palindrome_int = int(palindrome_str)
                        if palindrome_int % k == 0:
                            digit_counts = collections.Counter(palindrome_str)
                            num_permutations = self.count_permutations(n, digit_counts)
                            count += num_permutations
        else:
            m = n // 2
            for d1 in range(1, 10):
                for prefix_digits in self.generate_digit_sequences(m - 1, 0):
                    first_half = [d1] + prefix_digits
                    palindrome_digits = first_half + list(reversed(first_half))
                    palindrome_str = "".join(map(str, palindrome_digits))
                    palindrome_int = int(palindrome_str)
                    if palindrome_int % k == 0:
                        digit_counts = collections.Counter(palindrome_str)
                        num_permutations = self.count_permutations(n, digit_counts)
                        count += num_permutations
        return count

    def generate_digit_sequences(self, length, start_digit):
        if length == 0:
            yield []
        else:
            for digit in range(start_digit, 10):
                for sequence in self.generate_digit_sequences(length - 1, 0):
                    yield [digit] + sequence

    def count_permutations(self, n, digit_counts):
        total_permutations = self.factorial(n)
        denominator = 1
        for digit in digit_counts:
            denominator *= self.factorial(digit_counts[digit])
        result = total_permutations // denominator
        if '0' in digit_counts and digit_counts['0'] > 0:
            leading_zero_permutations_numerator = self.factorial(n - 1)
            leading_zero_permutations_denominator = 1
            digit_counts_no_leading_zero = digit_counts.copy()
            digit_counts_no_leading_zero['0'] -= 1
            for digit in digit_counts_no_leading_zero:
                if digit_counts_no_leading_zero[digit] > 0:
                    leading_zero_permutations_denominator *= self.factorial(digit_counts_no_leading_zero[digit])
            result -= leading_zero_permutations_numerator // leading_zero_permutations_denominator
        return result

    def factorial(self, n):
        if n <= 1:
            return 1
        else:
            res = 1
            for i in range(2, n + 1):
                res *= i
            return res