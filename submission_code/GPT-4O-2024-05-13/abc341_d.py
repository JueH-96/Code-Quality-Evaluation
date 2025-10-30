# YOUR CODE HERE
import math

def kth_smallest_divisible(N, M, K):
    lcm_nm = (N * M) // math.gcd(N, M)
    
    def count_divisibles(x):
        return (x // N) + (x // M) - 2 * (x // lcm_nm)
    
    low, high = 1, 10**18
    while low < high:
        mid = (low + high) // 2
        if count_divisibles(mid) < K:
            low = mid + 1
        else:
            high = mid
    
    return low

import sys
input = sys.stdin.read
N, M, K = map(int, input().strip().split())
print(kth_smallest_divisible(N, M, K))