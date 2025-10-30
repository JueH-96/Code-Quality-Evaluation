import math

def find_kth_number():
    n, m, k = map(int, input().split())
    g = math.gcd(n, m)
    lcm = n * m // g
    low = 1
    high = 10**18  # A sufficiently large upper bound
    
    while low < high:
        mid = (low + high) // 2
        # Calculate the count of numbers <= mid divisible by exactly one of n or m
        cnt = (mid // n) + (mid // m) - 2 * (mid // lcm)
        if cnt < k:
            low = mid + 1
        else:
            high = mid
    print(low)

find_kth_number()