import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible(n, m, k):
    divisible_n = (k + n - 1) // n
    divisible_m = (k + m - 1) // m
    divisible_both = (k + lcm(n, m) - 1) // lcm(n, m)
    return divisible_n + divisible_m - 2 * divisible_both

def find_kth_divisible(n, m, k):
    left, right = 1, min(n, m) * k
    while left < right:
        mid = (left + right) // 2
        if count_divisible(n, m, mid) < k:
            left = mid + 1
        else:
            right = mid
    return left

n, m, k = map(int, input().split())
print(find_kth_divisible(n, m, k))