import math

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def solve():
    n, m, k = map(int, input().split())
    lcm_val = (n * m) // gcd(n, m)
    
    low = 1
    high = 2 * (n if n > m else m) * k
    
    ans = -1
    while low <= high:
        mid = (low + high) // 2
        
        count_n = mid // n
        count_m = mid // m
        count_lcm = mid // lcm_val
        
        count_only_n = count_n - count_lcm
        count_only_m = count_m - count_lcm
        
        count_total = count_only_n + count_only_m
        
        if count_total >= k:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    
    print(ans)

solve()