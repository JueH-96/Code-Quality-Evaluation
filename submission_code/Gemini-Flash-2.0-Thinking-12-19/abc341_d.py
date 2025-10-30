def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def solve():
    n, m, k = map(int, input().split())
    if n > m:
        n, m = m, n
    l = lcm(n, m)
    
    def count_divisible_exactly_one(x):
        return x // n + x // m - 2 * (x // l)
        
    low = 1
    high = 2 * 10**18  # Sufficiently large upper bound
    result = -1
    
    while low <= high:
        mid = (low + high) // 2
        current_count = count_divisible_exactly_one(mid)
        if current_count >= k:
            result = mid
            high = mid - 1
        else:
            low = mid + 1
            
    print(result)

if __name__ == '__main__':
    solve()