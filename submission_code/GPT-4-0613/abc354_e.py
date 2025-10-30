import sys
from collections import Counter
from heapq import heapify, heappop, heappush

def solve():
    N = int(sys.stdin.readline())
    cards = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    cards.sort()
    cards = cards[::-1]
    heap = []
    for a, b in cards:
        while heap and heap[0] < a:
            heappop(heap)
        heappush(heap, b)
    print('Takahashi' if len(heap) % 2 == 1 else 'Aoki')

solve()