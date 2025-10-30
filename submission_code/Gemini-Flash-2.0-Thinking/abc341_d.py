def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    n, m, k = map(int, input().split())

    lcm_val = (n * m) // gcd(n, m)

    def count(x):
        return x // n + x // m - 2 * (x // lcm_val)

    low = 1
    high = 2 * 10**15  # A sufficiently large upper bound
    ans = high

    while low <= high:
        mid = (low + high) // 2
        if count(mid) >= k:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1

    print(ans)

solve()