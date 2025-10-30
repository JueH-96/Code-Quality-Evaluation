def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

def solve():
    N = int(input())
    
    # Find cube root of N and round down
    max_x = int(pow(N, 1/3))
    
    # Check cubes from max possible down to 1
    for x in range(max_x, 0, -1):
        cube = x * x * x
        if cube <= N and is_palindrome(cube):
            print(cube)
            return

solve()