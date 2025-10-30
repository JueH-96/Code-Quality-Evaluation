# YOUR CODE HERE
def find_nth_palindrome(N):
    # Start with the smallest palindromes
    length = 1
    count = 9  # There are 9 palindromes of length 1 (1 to 9)
    
    # Special case for N = 1, which is 0
    if N == 1:
        return 0
    
    # Adjust N because we are considering 0 as the first palindrome
    N -= 1
    
    # Find the length of the palindrome that contains the N-th palindrome
    while N > count:
        N -= count
        length += 1
        if length % 2 == 0:
            count = 9 * (10 ** ((length // 2) - 1))
        else:
            count = 9 * (10 ** (length // 2))
    
    # Now we know the length of the palindrome
    half_length = (length + 1) // 2
    start = 10 ** (half_length - 1)
    number = start + (N - 1)
    
    # Construct the palindrome
    if length % 2 == 0:
        # Even length palindrome
        palindrome = str(number) + str(number)[::-1]
    else:
        # Odd length palindrome
        palindrome = str(number) + str(number)[-2::-1]
    
    return int(palindrome)

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Find and print the N-th smallest palindrome
print(find_nth_palindrome(N))