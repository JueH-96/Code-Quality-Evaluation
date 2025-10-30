import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    W = int(data[idx+1])
    X = int(data[idx+2])
    idx +=3
    
    P = int(data[idx])
    Q = int(data[idx+1])
    p = P-1
    q = Q-1
    idx +=2
    
    S = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        S.append(row)
        idx += W
    
    absorbed = [[False]*W for _ in range(H)]
    absorbed[p][q] = True
    current_strength = S[p][q]
    heap = []
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    for di, dj in directions:
        ni = p + di
        nj = q + dj
        if 0 <= ni < H and 0 <= nj < W and not absorbed[ni][nj]:
            heapq.heappush(heap, (-S[ni][nj], ni, nj))
    
    while heap:
        neg_s, i, j = heapq.heappop(heap)
        s = -neg_s
        if absorbed[i][j]:
            continue
        if s * X < current_strength:
            current_strength += s
            absorbed[i][j] = True
            for di, dj in directions:
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W and not absorbed[ni][nj]:
                    heapq.heappush(heap, (-S[ni][nj], ni, nj))
    
    print(current_strength)

if __name__ == "__main__":
    main()