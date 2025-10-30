n = int(input())

low = 1
high = 10**6
x_max = 1

# Binary search to find the largest x where x^3 <= n
while low <= high:
    mid = (low + high) // 2
    cube = mid ** 3
    if cube <= n:
        x_max = mid
        low = mid + 1
    else:
        high = mid - 1

# Iterate from x_max down to 1 to find the largest palindromic cube
for x in range(x_max, 0, -1):
    cube = x ** 3
    s = str(cube)
    if s == s[::-1]:
        print(cube)
        break