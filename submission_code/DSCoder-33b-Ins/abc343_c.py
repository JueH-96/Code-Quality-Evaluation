# YOUR CODE HERE

def is_palindrome(n):
    return str(n) == str(n)[::-1]

def max_palindromic_cube(n):
    i = 1
    while i**3 <= n:
        if is_palindrome(i**3):
            max_palindrome = i**3
        i += 1
    return max_palindrome

n = int(input())
print(max_palindromic_cube(n))