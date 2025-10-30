import sys
import heapq
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A = deque(A)
    A.append(N+1)
    h = []
    for i in range(N, 0, -1):
        if A[-1] == i:
            A.pop()
            heapq.heappush(h, -i)
        else:
            if len(h) > 0 and -h[0] < i:
                heapq.heappop(h)
                heapq.heappush(h, -i)
            heapq.heappush(h, -i)
    ans = 0
    while len(h) > 1:
        ans += heapq.heappop(h)
    print(-ans)

if __name__ == "__main__":
    main()