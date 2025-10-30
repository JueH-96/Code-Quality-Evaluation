# YOUR CODE HERE
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

n, m, k = map(int, input().split())

lcm_nm = lcm(n, m)
count_n = k // n
count_m = k // m
count_both = k // lcm_nm

ans = k + (k - count_n - count_m + count_both)

print(ans)