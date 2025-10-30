import sys
from heapq import *

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    st = [list(map(int, sys.stdin.readline().split())) for _ in range(n-1)]
    st.append([0, 0])
    st.reverse()
    a.reverse()
    q = []
    for i in range(n):
        heappush(q, st[i][1])
        while len(q) > 0 and q[0] < a[i]:
            heappop(q)
        if len(q) > 0:
            a[i] -= q[0]
            heappush(q, st[i][0])
        else:
            a[i] = 0
    print(sum(a))

main()