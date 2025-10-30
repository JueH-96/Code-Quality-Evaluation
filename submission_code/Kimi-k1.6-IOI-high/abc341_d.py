import math

def find_kth_number():
    N, M, K = map(int, input().split())
    gcd_val = math.gcd(N, M)
    lcm = N * M // gcd_val
    low = 1
    high = 2 * 10**18  # Sufficiently large upper bound
    
    def count(x):
        return (x // N) + (x // M) - 2 * (x // lcm)
    
    while low < high:
        mid = (low + high) // 2
        c = count(mid)
        if c < K:
            low = mid + 1
        else:
            high = mid
    print(low)

find_kth_number()