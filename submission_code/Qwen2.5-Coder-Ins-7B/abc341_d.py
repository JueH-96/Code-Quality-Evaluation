# YOUR CODE HERE
import sys

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisibles(n, m, k):
    lcm_nm = lcm(n, m)
    count_n = k // n
    count_m = k // m
    count_lcm_nm = k // lcm_nm
    return count_n + count_m - count_lcm_nm

def find_kth_divisible(n, m, k):
    low, high = 1, k * min(n, m)
    while low < high:
        mid = (low + high) // 2
        if count_divisibles(n, m, mid) < k:
            low = mid + 1
        else:
            high = mid
    return low

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
K = int(data[2])

# Find and print the K-th smallest positive integer divisible by exactly one of N and M
result = find_kth_divisible(N, M, K)
print(result)