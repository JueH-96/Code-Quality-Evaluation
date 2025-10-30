import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    freq = defaultdict(int)
    for num in A:
        freq[num] += 1
    
    max_unique = -1
    max_val = -1
    for i in range(N):
        num = A[i]
        if freq[num] == 1:
            if num > max_val:
                max_val = num
                max_unique = i + 1  # since labels are 1-based
    
    print(max_unique)

solve()