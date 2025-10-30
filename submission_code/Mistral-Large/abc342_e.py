import sys
from heapq import heappop, heappush

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    trains = []

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

        for i in range(k):
            t = l + i * d
            trains.append((t, t + c, A, B))

    trains.sort(reverse=True)

    latest_arrival = [None] * (N + 1)
    latest_arrival[N] = float('inf')

    heap = []

    for t, t_end, A, B in trains:
        if latest_arrival[B] is not None:
            heappush(heap, (t_end, B))

        while heap and heap[0][0] < t:
            _, station = heappop(heap)
            if latest_arrival[station] == t:
                latest_arrival[station] = None

        if latest_arrival[A] is not None:
            latest_arrival[A] = t

    for i in range(1, N):
        if latest_arrival[i] is None:
            print("Unreachable")
        else:
            print(latest_arrival[i])

if __name__ == "__main__":
    main()