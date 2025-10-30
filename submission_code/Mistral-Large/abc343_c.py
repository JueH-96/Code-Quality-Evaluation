import sys

def is_palindrome(s):
    return s == s[::-1]

def find_max_palindromic_cube(N):
    limit = int(N**(1/3)) + 1
    max_palindromic_cube = -1

    for x in range(limit, 0, -1):
        cube = x**3
        if cube <= N and is_palindrome(str(cube)):
            max_palindromic_cube = cube
            break

    return max_palindromic_cube

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    result = find_max_palindromic_cube(N)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()