def kth_smallest_divisible(N, M, K):
    def count_divisibles(x):
        return (x // N) + (x // M) - (x // (N * M // gcd(N, M)))

    left, right = 1, 2 * K * max(N, M)
    
    while left < right:
        mid = (left + right) // 2
        if count_divisibles(mid) < K:
            left = mid + 1
        else:
            right = mid
            
    return left

import sys
from math import gcd

input_data = sys.stdin.read().strip()
N, M, K = map(int, input_data.split())
result = kth_smallest_divisible(N, M, K)
print(result)