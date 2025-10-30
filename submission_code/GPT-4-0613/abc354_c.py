import sys
from operator import itemgetter
from heapq import heappop, heappush

def main():
    N = int(sys.stdin.readline())
    cards = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    cards = sorted([(a, -c, i+1) for i, (a, c) in enumerate(cards)], key=itemgetter(0, 1))

    heap = []
    ans = []
    for a, c, i in cards:
        while heap and heap[0][0] > c:
            _, j = heappop(heap)
            ans.append(j)
        heappush(heap, (c, i))

    while heap:
        _, j = heappop(heap)
        ans.append(j)

    print(len(ans))
    print(*ans)

if __name__ == "__main__":
    main()