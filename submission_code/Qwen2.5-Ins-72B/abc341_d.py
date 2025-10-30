# YOUR CODE HERE
import sys

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible_by_one_of(n, m, x):
    return (x // n + x // m - 2 * (x // lcm(n, m)))

def find_kth_smallest(n, m, k):
    low, high = 1, 10**18
    while low < high:
        mid = (low + high) // 2
        if count_divisible_by_one_of(n, m, mid) < k:
            low = mid + 1
        else:
            high = mid
    return low

input = sys.stdin.read
N, M, K = map(int, input().split())
print(find_kth_smallest(N, M, K))