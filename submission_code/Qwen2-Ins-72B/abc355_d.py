import sys
from heapq import heappush, heappop

input = sys.stdin.readline

N = int(input())
L = []
R = []
for _ in range(N):
    l, r = map(int, input().split())
    L.append(l)
    R.append(r)

ans = 0
heap = []
for i, l in enumerate(L):
    while heap and heap[0] < l:
        ans += i - heap.pop()
    heappush(heap, R[i])
print(ans)