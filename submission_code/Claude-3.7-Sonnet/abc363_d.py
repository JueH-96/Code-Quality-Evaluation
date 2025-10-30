def nth_palindrome(N):
    # Handle 1-digit palindromes (0-9)
    if N <= 10:
        return N - 1
    
    N -= 10  # Adjust for the 10 1-digit palindromes
    
    # Handle 2-digit palindromes (11, 22, ..., 99)
    if N <= 9:
        return 11 * N
    
    N -= 9  # Adjust for the 9 2-digit palindromes
    
    # Find the digit length for larger palindromes
    digit_length = 3
    cumulative = 19  # Cumulative count after 1-digit and 2-digit palindromes
    
    while True:
        # Number of palindromes with current digit length
        count = 9 * 10**((digit_length - 1) // 2)
        if N <= count:
            break
        N -= count
        digit_length += 1
    
    # Calculate the "half" of the palindrome
    half = 10**((digit_length - 1) // 2) + (N - 1)
    
    # Convert to string for easier manipulation
    half_str = str(half)
    
    # For odd digit lengths, exclude the last digit from reflection
    if digit_length % 2 == 1:
        first_half = half_str[:-1]
        middle = half_str[-1]
        second_half = first_half[::-1]
        palindrome_str = first_half + middle + second_half
    else:
        first_half = half_str
        second_half = first_half[::-1]
        palindrome_str = first_half + second_half
    
    return int(palindrome_str)

# Read input
N = int(input())
print(nth_palindrome(N))