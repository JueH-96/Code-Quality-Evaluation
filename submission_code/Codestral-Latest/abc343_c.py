# YOUR CODE HERE
import sys

def is_palindrome(s):
    return s == s[::-1]

def find_max_palindromic_cube(N):
    x = int(N ** (1/3)) + 1
    while x > 0:
        cube = x ** 3
        if cube > N:
            x -= 1
            continue
        if is_palindrome(str(cube)):
            return cube
        x -= 1
    return None

def main():
    input = sys.stdin.read
    N = int(input())
    result = find_max_palindromic_cube(N)
    print(result)

if __name__ == "__main__":
    main()