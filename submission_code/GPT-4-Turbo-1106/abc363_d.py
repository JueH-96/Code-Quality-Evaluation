def find_nth_palindrome(n):
    if n == 1:
        return 0
    n -= 1
    digits = 1
    while True:
        if digits % 2 == 0:
            palindromes_with_digits = 9 * (10 ** (digits // 2 - 1))
        else:
            palindromes_with_digits = 10 ** (digits // 2)
        if n <= palindromes_with_digits:
            break
        n -= palindromes_with_digits
        digits += 1
    half = str(10 ** (digits // 2 - 1) + n - 1)
    if digits % 2 == 0:
        return int(half + half[::-1])
    else:
        return int(half + half[-2::-1])

# Read input from stdin
N = int(input().strip())

# Find and print the N-th smallest palindrome number
print(find_nth_palindrome(N))