def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def count_exactly_one_divisible(x, n, m, l):
    return x // n + x // m - 2 * (x // l)

def solve():
    n, m, k = map(int, input().split())
    l = lcm(n, m)
    low = 1
    high = 2 * k * max(n, m)
    result = -1
    while low <= high:
        mid = (low + high) // 2
        count = count_exactly_one_divisible(mid, n, m, l)
        if count >= k:
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    print(result)

if __name__ == '__main__':
    solve()