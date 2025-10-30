# YOUR CODE HERE
def count_palindromes_with_digits(d):
    """Count palindromes with exactly d digits"""
    if d == 1:
        return 10
    elif d % 2 == 0:
        return 9 * (10 ** (d // 2 - 1))
    else:
        return 9 * (10 ** ((d - 1) // 2))

def nth_palindrome(n):
    """Find the n-th smallest palindrome (1-indexed)"""
    if n <= 10:
        return n - 1
    
    # Find the number of digits
    count = 10  # palindromes with 1 digit
    digits = 1
    
    while count < n:
        digits += 1
        palindromes_with_d_digits = count_palindromes_with_digits(digits)
        if count + palindromes_with_d_digits >= n:
            break
        count += palindromes_with_d_digits
    
    # Now we know the n-th palindrome has 'digits' digits
    # Find which one it is among palindromes with 'digits' digits
    position = n - count - 1  # 0-indexed position among palindromes with 'digits' digits
    
    if digits % 2 == 0:
        # Even number of digits
        half_digits = digits // 2
        # The first half determines the palindrome
        first_half = 10 ** (half_digits - 1) + position
        palindrome_str = str(first_half) + str(first_half)[::-1]
    else:
        # Odd number of digits
        half_digits = (digits - 1) // 2
        # The first half + middle digit determines the palindrome
        first_half_and_middle = 10 ** half_digits + position
        first_half_and_middle_str = str(first_half_and_middle)
        palindrome_str = first_half_and_middle_str + first_half_and_middle_str[-2::-1]
    
    return int(palindrome_str)

n = int(input())
print(nth_palindrome(n))