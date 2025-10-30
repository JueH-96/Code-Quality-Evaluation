import sys
from bisect import bisect_left, bisect_right

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    A = sorted(map(int, sys.stdin.readline().split()))
    return N, M, A

def solve():
    N, M, A = read_input()
    ans = 0
    for a in A:
        l = bisect_left(A, a)
        r = bisect_right(A, a+M-1)
        if r - l > ans:
            ans = r - l
    print(ans)

solve()