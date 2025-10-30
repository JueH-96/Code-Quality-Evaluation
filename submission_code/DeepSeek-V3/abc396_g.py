import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Check for adjacent duplicates first
    for i in range(N - 1):
        if A[i] == A[i + 1]:
            print(2)
            return
    
    last_pos = {}
    min_len = float('inf')
    
    for idx, num in enumerate(A):
        if num in last_pos:
            current_len = idx - last_pos[num] + 1
            if current_len < min_len:
                min_len = current_len
                if min_len == 3:  # can't get smaller than this for non-adjacent
                    break
        last_pos[num] = idx
    
    if min_len != float('inf'):
        print(min_len)
    else:
        print(-1)

solve()