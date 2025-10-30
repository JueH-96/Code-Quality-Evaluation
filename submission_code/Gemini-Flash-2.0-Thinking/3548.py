from collections import Counter
from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        count = 0

        def get_permutations(counts):
            n_digits = sum(counts.values())
            numerator = factorial(n_digits)
            denominator = 1
            for c in counts.values():
                denominator *= factorial(c)
            return numerator // denominator

        def count_arrangements(counts):
            total_permutations = get_permutations(counts)
            if 0 in counts and counts[0] > 0:
                counts_without_leading_zero = counts.copy()
                counts_without_leading_zero[0] -= 1
                permutations_with_leading_zero = get_permutations(counts_without_leading_zero) if sum(counts_without_leading_zero.values()) == n - 1 else 0
                return total_permutations - permutations_with_leading_zero
            else:
                return total_permutations

        def can_form_palindrome(counts):
            odd_counts = sum(1 for c in counts.values() if c % 2 != 0)
            return odd_counts <= 1 if n % 2 != 0 else odd_counts == 0

        def check_palindrome_divisibility(counts):
            def generate_palindromes_from_counts(current_counts, current_palindrome):
                if len(current_palindrome) == n:
                    return [int("".join(map(str, current_palindrome)))]

                palindromes = []
                remaining_counts = current_counts.copy()

                for digit in range(10):
                    if remaining_counts.get(digit, 0) > 0:
                        remaining_counts[digit] -= 1
                        palindromes.extend(generate_palindromes_from_counts(remaining_counts, current_palindrome + [digit]))
                        remaining_counts[digit] += 1
                return palindromes

            from itertools import permutations

            digits = []
            for digit, c in counts.items():
                digits.extend([digit] * c)

            for perm in set(permutations(digits)):
                if perm[0] == 0 and n > 1:
                    continue
                num_str = "".join(map(str, perm))
                num = int(num_str)
                if str(num) == num_str and self.is_palindrome(num) and num % k == 0:
                    return True
            return False

        def find_good_counts(index, current_counts):
            if index == 10:
                if sum(current_counts) == n:
                    counts_dict = {i: current_counts[i] for i in range(10) if current_counts[i] > 0}
                    if can_form_palindrome(counts_dict) and check_palindrome_divisibility(counts_dict):
                        nonlocal count
                        count += count_arrangements(counts_dict)
                return

            if sum(current_counts) > n:
                return

            find_good_counts(index + 1, current_counts) # Skip current digit count

            current_counts[index] += 1
            find_good_counts(index, current_counts)
            current_counts[index] -= 1

        find_good_counts(0, [0] * 10)
        return count

    def is_palindrome(self, n):
        return str(n) == str(n)[::-1]