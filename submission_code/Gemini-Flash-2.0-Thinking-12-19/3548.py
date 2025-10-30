from collections import Counter

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        count = 0
        digits = list(range(10))
        
        def get_digit_counts(n_digits):
            counts_list = []
            def generate_counts(index, remaining_sum, current_counts):
                if index == 10:
                    if remaining_sum == 0:
                        counts_list.append(tuple(current_counts))
                    return
                for i in range(remaining_sum + 1):
                    current_counts[index] = i
                    generate_counts(index + 1, remaining_sum - i, current_counts)
            
            initial_counts = [0] * 10
            generate_counts(0, n_digits, initial_counts)
            return counts_list
            
        all_digit_counts = get_digit_counts(n)
        valid_digit_counts_list = []
        for digit_counts_tuple in all_digit_counts:
            digit_counts = list(digit_counts_tuple)
            odd_counts = 0
            for c in digit_counts:
                if c % 2 != 0:
                    odd_counts += 1
            if n % 2 == 0:
                if odd_counts == 0:
                    valid_digit_counts_list.append(digit_counts)
            else:
                if odd_counts == 1:
                    valid_digit_counts_list.append(digit_counts)
                    
        good_integer_count = 0
        
        for digit_counts in valid_digit_counts_list:
            found_k_palindrome = False
            
            def generate_palindromes(current_prefix_digits, remaining_counts):
                nonlocal found_k_palindrome
                if found_k_palindrome:
                    return
                prefix_len = len(current_prefix_digits)
                if n % 2 == 0:
                    if prefix_len == n // 2:
                        palindrome_digits = current_prefix_digits + current_prefix_digits[::-1]
                        if palindrome_digits[0] == 0 and n > 1:
                            return
                        palindrome_num = 0
                        for digit in palindrome_digits:
                            palindrome_num = palindrome_num * 10 + digit
                        if palindrome_num % k == 0:
                            found_k_palindrome = True
                        return
                else:
                    if prefix_len == (n - 1) // 2:
                        for middle_digit in range(10):
                            if remaining_counts[middle_digit] > 0:
                                next_remaining_counts = list(remaining_counts)
                                next_remaining_counts[middle_digit] -= 1
                                palindrome_digits = current_prefix_digits + [middle_digit] + current_prefix_digits[::-1]
                                if palindrome_digits[0] == 0 and n > 1:
                                    continue
                                palindrome_num = 0
                                for digit in palindrome_digits:
                                    palindrome_num = palindrome_num * 10 + digit
                                if palindrome_num % k == 0:
                                    found_k_palindrome = True
                                if found_k_palindrome:
                                    return
                        return
                        
                for digit in range(10):
                    if remaining_counts[digit] >= 2:
                        next_prefix_digits = current_prefix_digits + [digit]
                        next_remaining_counts = list(remaining_counts)
                        next_remaining_counts[digit] -= 2
                        generate_palindromes(next_prefix_digits, next_remaining_counts)
                        if found_k_palindrome:
                            return
                            
            generate_palindromes([], digit_counts)
            
            if found_k_palindrome:
                total_permutations = math.factorial(n)
                leading_zero_permutations = 0
                if digit_counts[0] > 0 and n > 1:
                    leading_zero_permutations = math.factorial(n - 1)
                    
                denominator = 1
                for c in digit_counts:
                    denominator *= math.factorial(c)
                    
                permutations_count = total_permutations // denominator
                if digit_counts[0] > 0 and n > 1:
                    denominator_zero_leading = 1
                    digit_counts_zero_leading = list(digit_counts)
                    digit_counts_zero_leading[0] -= 1
                    for c in digit_counts_zero_leading:
                        denominator_zero_leading *= math.factorial(c)
                    permutations_count -= (leading_zero_permutations // denominator_zero_leading)
                    
                good_integer_count += permutations_count
                
        return good_integer_count

import math