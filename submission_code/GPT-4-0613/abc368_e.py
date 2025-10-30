import sys
from heapq import heappop, heappush

def main():
    n, m, x = map(int, sys.stdin.readline().split())
    trains = [list(map(int, sys.stdin.readline().split())) for _ in range(m)]
    trains.sort(key=lambda x: (x[1], x[3]))
    trains = [(a-1, b-1, s, t) for a, b, s, t in trains]

    delay = [0]*m
    heap = []
    j = 0
    for i in range(n):
        while j < m and trains[j][1] == i:
            while heap and heap[0][0] <= trains[j][2]:
                _, idx = heappop(heap)
                delay[idx] = max(delay[idx], trains[j][2] + x - trains[idx][3])
            heappush(heap, (trains[j][3], j))
            j += 1
        while heap:
            _, idx = heappop(heap)
            if heap:
                delay[idx] = max(delay[idx], heap[0][0] + x - trains[idx][3])

    print(*delay)

if __name__ == "__main__":
    main()