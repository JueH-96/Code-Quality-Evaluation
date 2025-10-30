import sys
from heapq import heapify, heappop, heappush

def main():
    N = int(sys.stdin.readline())
    FS = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    FS.sort(key=lambda x: (-x[1], x[0]))

    heap = []
    heapify(heap)
    flavor = -1
    max_satisfaction = 0
    for f, s in FS:
        while heap and heap[0][1] < f:
            _, flavor = heappop(heap)
        if flavor != -1:
            max_satisfaction = max(max_satisfaction, s + heap[0][0] // 2)
        heappush(heap, (-s, f))
        max_satisfaction = max(max_satisfaction, s + heap[0][0] // 2)
    print(max_satisfaction)

if __name__ == "__main__":
    main()