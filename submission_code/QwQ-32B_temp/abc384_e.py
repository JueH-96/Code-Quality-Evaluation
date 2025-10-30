import heapq

def main():
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    P -= 1
    Q -= 1

    S = []
    for _ in range(H):
        row = list(map(int, input().split()))
        S.append(row)

    visited = [[0 for _ in range(W)] for _ in range(H)]
    visited[P][Q] = 2  # mark as part of the region
    current_strength = S[P][Q]

    heap = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize border cells
    for di, dj in directions:
        ni = P + di
        nj = Q + dj
        if 0 <= ni < H and 0 <= nj < W:
            if visited[ni][nj] == 0:
                visited[ni][nj] = 1  # mark as border
                heapq.heappush(heap, (-S[ni][nj], ni, nj))

    while heap:
        neg_s, i, j = heapq.heappop(heap)
        s = -neg_s
        if visited[i][j] != 1:
            continue  # already processed or not in border
        if s * X < current_strength:
            current_strength += s
            visited[i][j] = 2  # absorb into region
            for di, dj in directions:
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if visited[ni][nj] == 0:
                        visited[ni][nj] = 1
                        heapq.heappush(heap, (-S[ni][nj], ni, nj))
        else:
            continue  # cannot absorb now

    print(current_strength)

if __name__ == "__main__":
    main()