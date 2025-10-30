def find_minimum_X(N, M, A, B):
    A.sort()
    B.sort()
    
    # Binary search for the minimum X
    low, high = 1, 10**9 + 1
    
    while low < high:
        mid = (low + high) // 2
        
        # Number of sellers who can sell at mid or more
        sellers_count = len(A) - bisect.bisect_left(A, mid)
        
        # Number of buyers who can buy at mid or less
        buyers_count = bisect.bisect_right(B, mid)
        
        if sellers_count >= buyers_count:
            high = mid
        else:
            low = mid + 1
    
    return low

import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    result = find_minimum_X(N, M, A, B)
    print(result)