import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    P -= 1
    Q -= 1

    S = [list(map(int, input().split())) for _ in range(H)]

    # visited marks cells that are either absorbed already or are in the candidate heap
    visited = [[False]*W for _ in range(H)]
    visited[P][Q] = True

    # current strength of Takahashi
    curr = S[P][Q]

    # min-heap of (strength, i, j) for frontier slimes
    heap = []
    def push(i, j):
        visited[i][j] = True
        heapq.heappush(heap, (S[i][j], i, j))

    # push the neighbors of the starting cell
    for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
        ni, nj = P+di, Q+dj
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
            push(ni, nj)

    # Try to absorb as long as the smallest candidate is strictly less than curr/X
    while heap:
        val, i, j = heap[0]
        # Can we absorb this slime?
        if val * X >= curr:
            break
        # Yes: absorb it
        heapq.heappop(heap)
        curr += val
        # Add its neighbors into the frontier
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
                push(ni, nj)

    print(curr)

if __name__ == "__main__":
    main()