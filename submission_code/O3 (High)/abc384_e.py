import sys, heapq

def main() -> None:
    it = iter(sys.stdin.read().split())
    H = int(next(it)); W = int(next(it)); X = int(next(it))
    P = int(next(it)) - 1          # 0-based
    Q = int(next(it)) - 1

    S = [[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            S[i][j] = int(next(it))

    visited = [[False]*W for _ in range(H)]
    visited[P][Q] = True
    cur = S[P][Q]                   # current strength

    heap = []
    def push(i: int, j: int) -> None:
        if 0 <= i < H and 0 <= j < W and not visited[i][j]:
            heapq.heappush(heap, (S[i][j], i, j))

    # initial boundary
    push(P-1, Q); push(P+1, Q); push(P, Q-1); push(P, Q+1)

    while heap:
        s, i, j = heap[0]
        if visited[i][j]:
            heapq.heappop(heap)
            continue
        if cur > X * s:             # strictly larger
            heapq.heappop(heap)
            visited[i][j] = True
            cur += s
            # add new boundary cells
            push(i-1, j); push(i+1, j); push(i, j-1); push(i, j+1)
        else:
            break

    print(cur)

if __name__ == "__main__":
    main()