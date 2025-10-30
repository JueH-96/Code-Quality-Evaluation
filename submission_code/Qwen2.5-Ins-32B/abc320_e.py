import heapq

def solve():
    N, M = map(int, input().split())
    noodles = [0] * N
    queue = []
    for _ in range(M):
        T, W, S = map(int, input().split())
        while queue and queue[0][0] <= T:
            _, idx = heapq.heappop(queue)
            heapq.heappush(queue, (T + S, idx))
        if queue:
            _, idx = heapq.heappop(queue)
            noodles[idx] += W
            heapq.heappush(queue, (T + S, idx))
    for i in range(N):
        print(noodles[i])

if __name__ == "__main__":
    solve()