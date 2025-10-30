import sys
from heapq import heappush, heappop

def main():
    N = int(sys.stdin.readline())
    XY = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    XY.sort(key=lambda x: (-x[0], -x[1]))

    q = []
    ans = 0
    for x, y in XY:
        if x == 1:
            if q and q[0] < y:
                ans += y - heappop(q)
            else:
                ans += y
        else:
            if q and q[0] < y:
                heappush(q, y)
                ans += y - heappop(q)
            else:
                heappush(q, y)
                ans += y
    print(ans)

if __name__ == "__main__":
    main()