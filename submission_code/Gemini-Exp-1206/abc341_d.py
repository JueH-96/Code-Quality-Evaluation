def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_numbers(n, m, k, x):
    count = x // n + x // m - 2 * (x // lcm(n, m))
    return count

def solve():
    n, m, k = map(int, input().split())
    
    left = 1
    right = 10**20
    
    while left <= right:
        mid = (left + right) // 2
        if count_numbers(n, m, k, mid) >= k:
            right = mid - 1
        else:
            left = mid + 1
    
    print(left)

solve()