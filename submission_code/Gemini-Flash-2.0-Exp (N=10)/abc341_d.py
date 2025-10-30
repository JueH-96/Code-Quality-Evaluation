def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible_by_only_one(n, m, x):
    l = lcm(n, m)
    count_n = x // n
    count_m = x // m
    count_l = x // l
    return count_n + count_m - 2 * count_l

def solve():
    n, m, k = map(int, input().split())
    
    low = 1
    high = 2 * 10**18
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        count = count_divisible_by_only_one(n, m, mid)
        if count >= k:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    print(ans)

solve()