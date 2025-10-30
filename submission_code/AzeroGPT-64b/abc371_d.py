import sys
from bisect import bisect_left, bisect_right

N = int(sys.stdin.readline())
X = list(map(int, sys.stdin.readline().split()))
P = list(map(int, sys.stdin.readline().split()))

Q = int(sys.stdin.readline())
for _ in range(Q):
    L, R = map(int, sys.stdin.readline().split())
    left = bisect_left(X, L)
    right = bisect_right(X, R)
    answer = sum(P[left:right])
    print(answer)