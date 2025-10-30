import sys
import heapq

def solve(N, K, Q, updates):
    A = [0]*N
    h = []
    result = 0
    for i in range(Q):
        x, y = updates[i]
        heapq.heappush(h, -y)
        A[x-1] = y
        if len(h) > K:
            result -= -heapq.heappop(h)
        print(result)

def main():
    N, K, Q = map(int, sys.stdin.readline().split())
    updates = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    solve(N, K, Q, updates)

if __name__ == "__main__":
    main()