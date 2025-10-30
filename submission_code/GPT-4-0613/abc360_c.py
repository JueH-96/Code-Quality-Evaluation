import sys
from heapq import heapify, heappop, heappush

def main():
    N = int(input())
    A = list(map(int, input().split()))
    W = list(map(int, input().split()))
    box = [[] for _ in range(N+1)]
    for i in range(N):
        box[A[i]].append(-W[i])
    for i in range(1, N+1):
        box[i].sort()
    heap = []
    ans = 0
    for i in range(1, N+1):
        if len(box[i]) == 0:
            w = -heappop(heap)
            ans += w
        else:
            while len(box[i]) > 1:
                w = -heappop(box[i])
                ans += w
                heappush(heap, -w)
            heappush(heap, box[i][0])
    print(ans)

if __name__ == "__main__":
    main()