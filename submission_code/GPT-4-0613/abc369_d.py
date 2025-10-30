import sys
from heapq import heapify, heappush, heappop

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    a.sort()
    q = []
    heapify(q)
    ans = 0
    for i in range(n):
        if len(q) < 2:
            heappush(q, a[i])
        else:
            if q[0] < a[i]:
                heappop(q)
                heappush(q, a[i])
        ans += len(q) * a[i]
    print(ans)

if __name__ == "__main__":
    main()