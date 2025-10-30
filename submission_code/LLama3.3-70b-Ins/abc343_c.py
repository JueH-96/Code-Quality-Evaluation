def is_palindrome(n):
    """Check if a number is a palindrome."""
    return str(n) == str(n)[::-1]

def find_max_palindromic_cube(N):
    """Find the maximum value of a palindromic cube number not greater than N."""
    max_palindromic_cube = 0
    i = 1
    while True:
        cube = i ** 3
        if cube > N:
            break
        if is_palindrome(cube):
            max_palindromic_cube = max(max_palindromic_cube, cube)
        i += 1
    return max_palindromic_cube

# Read input from stdin
N = int(input())

# Find and print the maximum palindromic cube
print(find_max_palindromic_cube(N))