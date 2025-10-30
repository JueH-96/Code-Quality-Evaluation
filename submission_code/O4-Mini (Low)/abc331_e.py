import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    N, M, L = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    # Keep original indices with values
    A = [ (a[i], i+1) for i in range(N) ]
    B = [ (b[j], j+1) for j in range(M) ]
    # Sort descending by price
    A.sort(key=lambda x: x[0], reverse=True)
    B.sort(key=lambda x: x[0], reverse=True)

    # Read forbidden pairs into a set
    forbidden = set()
    for _ in range(L):
        c, d = map(int, input().split())
        forbidden.add((c, d))

    # Max-heap over sums: we store negative sum to simulate max-heap
    # Each entry is ( - (A[x].price + B[y].price), x, y )
    heap = []
    visited = set()

    # start with the pair (0,0)
    initial_sum = A[0][0] + B[0][0]
    heapq.heappush(heap, (-initial_sum, 0, 0))
    visited.add((0, 0))

    while heap:
        neg_sum, x, y = heapq.heappop(heap)
        price_sum = -neg_sum
        # get original dish indices
        main_idx = A[x][1]
        side_idx = B[y][1]
        # if not forbidden, this is the max valid
        if (main_idx, side_idx) not in forbidden:
            print(price_sum)
            return
        # otherwise push neighbors (x+1, y) and (x, y+1)
        if x + 1 < N and (x+1, y) not in visited:
            visited.add((x+1, y))
            s = A[x+1][0] + B[y][0]
            heapq.heappush(heap, (-s, x+1, y))
        if y + 1 < M and (x, y+1) not in visited:
            visited.add((x, y+1))
            s = A[x][0] + B[y+1][0]
            heapq.heappush(heap, (-s, x, y+1))

if __name__ == "__main__":
    main()