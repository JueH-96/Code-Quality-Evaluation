# YOUR CODE HERE
import sys

def generate_palindromes(half, is_odd):
    if is_odd:
        return int(half + half[-2::-1])
    else:
        return int(half + half[::-1])

def find_nth_palindrome(n):
    if n == 1:
        return 0
    count = 1  # Starting with 0 as the first palindrome
    length = 1
    while True:
        start = 10 ** (length - 1)
        end = 10 ** length
        for i in range(start, end):
            half_length = (length + 1) // 2
            half = str(i)[:half_length]
            if length % 2 == 0:
                palindrome = generate_palindromes(half, False)
            else:
                palindrome = generate_palindromes(half, True)
            count += 1
            if count == n:
                return palindrome
        length += 1

n = int(sys.stdin.read().strip())
print(find_nth_palindrome(n))