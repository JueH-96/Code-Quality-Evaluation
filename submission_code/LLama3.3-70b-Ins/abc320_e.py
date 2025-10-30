import sys
import heapq

def solve():
    N, M = map(int, sys.stdin.readline().split())
    events = []
    for _ in range(M):
        T, W, S = map(int, sys.stdin.readline().split())
        events.append((T, W, S))

    events.sort()

    queue = list(range(1, N + 1))
    return_times = []
    noodles_got = [0] * N

    for T, W, S in events:
        while return_times and return_times[0][0] <= T:
            _, person = heapq.heappop(return_times)
            heapq.heappush(queue, person)

        if queue:
            person = heapq.heappop(queue)
            noodles_got[person - 1] += W
            if T + S <= 10**9:
                heapq.heappush(return_times, (T + S, person))

    for i in range(N):
        print(noodles_got[i])

if __name__ == "__main__":
    solve()