def find_nth_palindrome(N):
    if N == 1:
        return 0
    cumulative = 10  # Number of 1-digit palindromes
    if N <= cumulative:
        return N - 1  # Directly return 0 to 9
    N -= cumulative
    k = 2
    while True:
        half = (k + 1) // 2
        count = 9 * (10 ** (half - 1))
        if N <= count:
            first_half = 10 ** (half - 1) + (N - 1)
            first_str = str(first_half)
            if k % 2 == 0:
                palindrome_str = first_str + first_str[::-1]
            else:
                palindrome_str = first_str + first_str[:-1][::-1]
            return int(palindrome_str)
        N -= count
        k += 1

# Read input
N = int(input())
# Find and print the N-th smallest palindrome
print(find_nth_palindrome(N))