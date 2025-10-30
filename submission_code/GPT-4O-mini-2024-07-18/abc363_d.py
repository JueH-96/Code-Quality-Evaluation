def nth_palindrome(n):
    # Handle the case for the first palindrome
    if n == 1:
        return 0
    
    # Determine the number of digits in the palindrome
    length = 1
    count = 9  # There are 9 palindromes with 1 digit (0-9)
    total = count
    
    while total < n:
        length += 1
        count = 9 * (10 ** ((length - 1) // 2))  # Count of palindromes of this length
        total += count
    
    # Find the index of the palindrome in the current length
    index_in_length = n - (total - count) - 1
    
    # Generate the first half of the palindrome
    half_length = (length + 1) // 2
    first_half = str(10 ** (half_length - 1) + index_in_length)
    
    # Create the palindrome
    if length % 2 == 0:
        palindrome = first_half + first_half[::-1]
    else:
        palindrome = first_half + first_half[-2::-1]
    
    return palindrome

import sys
input = sys.stdin.read
N = int(input().strip())
print(nth_palindrome(N))