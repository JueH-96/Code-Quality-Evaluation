import sys

def find_x_max(n):
    low = 1
    high = 10**6
    best = 0
    while low <= high:
        mid = (low + high) // 2
        cube = mid ** 3
        if cube <= n:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

n = int(sys.stdin.readline())
x_max = find_x_max(n)

for x in range(x_max, 0, -1):
    cube = x ** 3
    s = str(cube)
    if s == s[::-1]:
        print(cube)
        break