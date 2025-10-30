import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    X = int(data[idx]); idx +=1
    P = int(data[idx]) -1; idx +=1
    Q = int(data[idx]) -1; idx +=1
    
    S = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        S.append(row)
        idx += W
    
    current_strength = S[P][Q]
    absorbed = [[False for _ in range(W)] for __ in range(H)]
    absorbed[P][Q] = True
    
    added = [[False for _ in range(W)] for __ in range(H)]
    
    priority_queue = []
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    for di, dj in directions:
        ni = P + di
        nj = Q + dj
        if 0 <= ni < H and 0 <= nj < W:
            if not absorbed[ni][nj] and not added[ni][nj]:
                s = S[ni][nj]
                s_x = s * X
                heapq.heappush(priority_queue, (s_x, s, ni, nj))
                added[ni][nj] = True
    
    while priority_queue:
        s_x, s, i, j = heapq.heappop(priority_queue)
        if current_strength <= s_x:
            break
        if absorbed[i][j]:
            continue
        current_strength += s
        absorbed[i][j] = True
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if not absorbed[ni][nj] and not added[ni][nj]:
                    adj_s = S[ni][nj]
                    adj_s_x = adj_s * X
                    heapq.heappush(priority_queue, (adj_s_x, adj_s, ni, nj))
                    added[ni][nj] = True
    
    print(current_strength)

if __name__ == '__main__':
    main()