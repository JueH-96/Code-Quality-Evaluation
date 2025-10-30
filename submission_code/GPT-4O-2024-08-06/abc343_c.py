# YOUR CODE HERE
def is_palindrome(num):
    s = str(num)
    return s == s[::-1]

def find_max_palindromic_cube(N):
    # Start from the largest possible cube root
    max_cube_root = int(N ** (1/3))
    
    # Iterate downwards to find the largest palindromic cube
    for x in range(max_cube_root, 0, -1):
        cube = x ** 3
        if cube <= N and is_palindrome(cube):
            return cube
    
    return 1  # If no palindromic cube is found, return 1 (since 1^3 = 1 is palindromic)

def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    result = find_max_palindromic_cube(N)
    print(result)

if __name__ == "__main__":
    main()