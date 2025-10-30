def is_palindrome(s):
    return s == s[::-1]

def solve(N):
    # Start from a reasonable upper bound for the cube root
    x = int(N**(1/3)) + 3  # Small buffer for precision issues
    
    # Iterate from x down to 1
    for i in range(x, 0, -1):
        cube = i**3
        if cube > N:
            continue
        if is_palindrome(str(cube)):
            return cube
    
    return 0  # This should never happen

N = int(input())
print(solve(N))