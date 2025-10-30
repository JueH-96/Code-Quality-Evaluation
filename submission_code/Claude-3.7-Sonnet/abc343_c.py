def is_palindrome(n):
    """Check if a number is a palindrome."""
    return str(n) == str(n)[::-1]

def find_max_palindromic_cube(N):
    """Find the maximum palindromic cube number not greater than N."""
    max_palindromic_cube = 0
    
    x = 1
    while x * x * x <= N:
        cube = x * x * x
        if is_palindrome(cube):
            max_palindromic_cube = cube
        x += 1
    
    return max_palindromic_cube

# Read input
N = int(input())

# Find and print the answer
print(find_max_palindromic_cube(N))