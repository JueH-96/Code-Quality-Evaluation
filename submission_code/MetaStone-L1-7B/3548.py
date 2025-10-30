import math
from collections import Counter

def countGoodIntegers(n, k):
    total = 0
    half = (n + 1) // 2
    for first_half in range(10 ** (half - 1), 10 ** half):
        s = str(first_half)
        if len(s) != half:
            continue  # Skip if length is incorrect (shouldn't happen for valid first_half)
        if n % 2 == 0:
            palindrome_str = s + s[::-1]
        else:
            palindrome_str = s + s[:-1][::-1]
        palindrome = int(palindrome_str)
        if palindrome % k == 0:
            digits = list(map(int, str(palindrome)))
            cnt = Counter(digits)
            total_digits = len(digits)
            # Compute total permutations
            numerator = math.factorial(total_digits)
            denominator = 1
            for v in cnt.values():
                denominator *= math.factorial(v)
            total_perm = numerator // denominator
            # Compute invalid permutations (starting with zero)
            invalid = 0
            if 0 in cnt:
                zero_cnt = cnt[0]
                if zero_cnt > 0:
                    temp = cnt.copy()
                    temp[0] -= 1
                    if temp[0] < 0:
                        temp[0] = 0
                    # Compute denominator for invalid permutations
                    invalid_denominator = 1
                    for v in temp.values():
                        invalid_denominator *= math.factorial(v)
                    invalid_perm = math.factorial(total_digits - 1) // invalid_denominator
                    invalid = invalid_perm
            count_numbers = total_perm - invalid
            total += count_numbers
    return total