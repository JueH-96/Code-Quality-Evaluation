import sys
from heapq import heappop, heappush

def main():
    N = int(sys.stdin.readline())
    bases = []
    for _ in range(N):
        W, X = map(int, sys.stdin.readline().split())
        bases.append((W, X))
    bases.sort(key=lambda x: x[1])
    heap = []
    ans = 0
    for W, X in bases:
        heappush(heap, (X + 9, W))
        while heap and heap[0][0] < X + 9:
            _, W2 = heappop(heap)
            ans -= W2
        ans += W
    print(ans)

if __name__ == "__main__":
    main()