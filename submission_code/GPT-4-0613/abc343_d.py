import sys
from collections import defaultdict
from heapq import heappop, heappush

def main():
    N, T = map(int, sys.stdin.readline().split())
    AB = [list(map(int, sys.stdin.readline().split())) for _ in range(T)]
    AB.sort()

    scores = defaultdict(int)
    scores[0] = N
    heap = []

    for a, b in AB:
        if heap and heap[0] < b:
            while heap and heap[0] < b:
                scores[heap[0]] -= 1
                if scores[heap[0]] == 0:
                    del scores[heap[0]]
                heappop(heap)
        scores[b] += 1
        heappush(heap, b)
        print(len(scores))

if __name__ == "__main__":
    main()