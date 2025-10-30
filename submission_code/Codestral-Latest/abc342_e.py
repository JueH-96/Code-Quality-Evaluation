import heapq
from collections import defaultdict

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    trains = defaultdict(list)

    for _ in range(M):
        l = int(data[index])
        index += 1
        d = int(data[index])
        index += 1
        k = int(data[index])
        index += 1
        c = int(data[index])
        index += 1
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1

        for t in range(l, l + k * d, d):
            trains[A].append((t, t + c, B))

    for station in trains:
        trains[station].sort()

    def latest_arrival_time(start):
        pq = []
        for t, arrival, dest in trains[start]:
            heapq.heappush(pq, (-arrival, dest))

        max_time = -1
        while pq:
            arrival, dest = heapq.heappop(pq)
            arrival = -arrival
            if dest == N:
                max_time = max(max_time, arrival)
            else:
                for t, next_arrival, next_dest in trains[dest]:
                    if t > arrival:
                        heapq.heappush(pq, (-next_arrival, next_dest))

        return max_time

    results = []
    for i in range(1, N):
        result = latest_arrival_time(i)
        if result == -1:
            results.append("Unreachable")
        else:
            results.append(result)

    for result in results:
        print(result)

solve()