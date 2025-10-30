# YOUR CODE HERE
import sys

def is_palindrome(s):
    return s == s[::-1]

def find_max_palindromic_cube(N):
    # Start from the cube root of N and go downwards
    x = int(N ** (1/3))
    while x > 0:
        cube = x ** 3
        if cube <= N and is_palindrome(str(cube)):
            return cube
        x -= 1
    return 1  # The smallest palindromic cube is 1 (1^3)

def main():
    N = int(sys.stdin.read().strip())
    result = find_max_palindromic_cube(N)
    print(result)

if __name__ == "__main__":
    main()