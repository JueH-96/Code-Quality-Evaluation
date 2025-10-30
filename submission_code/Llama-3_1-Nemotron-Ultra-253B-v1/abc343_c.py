n = int(input())

# Binary search to find the maximum x where x^3 <= N
low = 1
high = 10**6 + 1  # Sufficient upper bound as (1e6)^3 = 1e18
x_max = 0
while low <= high:
    mid = (low + high) // 2
    cube = mid ** 3
    if cube <= n:
        x_max = mid
        low = mid + 1
    else:
        high = mid - 1

# Check from x_max down to 1 for the largest palindromic cube
for x in range(x_max, 0, -1):
    cube = x ** 3
    s = str(cube)
    if s == s[::-1]:
        print(cube)
        break