import heapq
import sys
input = sys.stdin.readline

def solve():
    N, K = map(int, input().split())
    AB = [list(map(int, input().split())) for _ in range(N)]
    AB.sort(key=lambda x: -x[1])
    q = []
    ans = sum([b for a, b in AB[:K]])
    for a, b in AB[:K]:
        heapq.heappush(q, a)
    for a, b in AB[K:]:
        if a > q[0]:
            ans += a - heapq.heappop(q)
            heapq.heappush(q, a)
    print(ans)

if __name__ == "__main__":
    solve()