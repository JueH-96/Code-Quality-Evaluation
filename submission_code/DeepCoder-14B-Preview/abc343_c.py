import sys

def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

def find_x_max(n):
    if n < 1:
        return 0
    low = 1
    high = 1
    while high ** 3 <= n:
        high *= 2
    while low <= high:
        mid = (low + high) // 2
        cube = mid ** 3
        if cube == n:
            return mid
        elif cube < n:
            low = mid + 1
        else:
            high = mid - 1
    return high

n = int(sys.stdin.readline())
x_max = find_x_max(n)

for x in range(x_max, 0, -1):
    cube = x ** 3
    if is_palindrome(cube):
        print(cube)
        sys.exit()