N = int(input())

def find_max_x(n):
    low = 1
    high = 10**6
    best = 0
    while low <= high:
        mid = (low + high) // 2
        mid_cubed = mid ** 3
        if mid_cubed <= n:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

x_max = find_max_x(N)

for x in range(x_max, 0, -1):
    cube = x ** 3
    s = str(cube)
    if s == s[::-1]:
        print(cube)
        break