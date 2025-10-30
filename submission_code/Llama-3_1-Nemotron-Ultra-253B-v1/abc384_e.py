import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    X = int(input[idx]); idx +=1
    P = int(input[idx])-1; idx +=1
    Q = int(input[idx])-1; idx +=1
    
    S = []
    for _ in range(H):
        row = list(map(int, input[idx:idx+W]))
        idx += W
        S.append(row)
    
    current = S[P][Q]
    visited = [[False]*W for _ in range(H)]
    visited[P][Q] = True
    in_heap = [[False]*W for _ in range(H)]
    
    heap = []
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    for di, dj in dirs:
        ni = P + di
        nj = Q + dj
        if 0 <= ni < H and 0 <= nj < W:
            if not visited[ni][nj] and not in_heap[ni][nj]:
                s = S[ni][nj]
                heapq.heappush(heap, (s * X, s, ni, nj))
                in_heap[ni][nj] = True
    
    while heap:
        s_x, s, i, j = heapq.heappop(heap)
        if not in_heap[i][j]:
            continue
        in_heap[i][j] = False
        if current > s_x:
            current += s
            visited[i][j] = True
            for di, dj in dirs:
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if not visited[ni][nj] and not in_heap[ni][nj]:
                        new_s = S[ni][nj]
                        heapq.heappush(heap, (new_s * X, new_s, ni, nj))
                        in_heap[ni][nj] = True
        else:
            break
    
    print(current)

if __name__ == "__main__":
    main()