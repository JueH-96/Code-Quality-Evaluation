import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    A = int(input[idx]); idx += 1
    B = int(input[idx]); idx += 1
    C = int(input[idx]); idx += 1

    D = []
    for _ in range(N):
        row = list(map(int, input[idx:idx+N]))
        idx += N
        D.append(row)

    INF = float('inf')
    dist = [[INF] * 2 for _ in range(N + 1)]
    dist[1][0] = 0  # Starting at city 1 with car

    heap = []
    heapq.heappush(heap, (0, 1, 0))

    while heap:
        time, city, mode = heapq.heappop(heap)
        if time > dist[city][mode]:
            continue

        if mode == 0:
            # Take car to other cities
            for j in range(1, N + 1):
                if j == city:
                    continue
                new_time = time + D[city - 1][j - 1] * A
                if new_time < dist[j][0]:
                    dist[j][0] = new_time
                    heapq.heappush(heap, (new_time, j, 0))
            # Switch to train
            if time < dist[city][1]:
                dist[city][1] = time
                heapq.heappush(heap, (time, city, 1))
        else:
            # Take train to other cities
            for j in range(1, N + 1):
                if j == city:
                    continue
                new_time = time + D[city - 1][j - 1] * B + C
                if new_time < dist[j][1]:
                    dist[j][1] = new_time
                    heapq.heappush(heap, (new_time, j, 1))

    result = min(dist[N][0], dist[N][1])
    print(result if result != INF else -1)

if __name__ == '__main__':
    main()