import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    X = int(data[idx]); idx +=1
    P = int(data[idx]); idx +=1
    Q = int(data[idx]); idx +=1
    P -= 1
    Q -= 1
    S = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        S.append(row)
    
    current_S = S[P][Q]
    absorbed = [[False for _ in range(W)] for _ in range(H)]
    absorbed[P][Q] = True
    
    # Directions: up, down, left, right
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    adjacent = set()
    for dx, dy in dirs:
        ni = P + dx
        nj = Q + dy
        if 0 <= ni < H and 0 <= nj < W:
            adjacent.add((ni, nj))
    
    heap = []
    for i, j in adjacent:
        if not absorbed[i][j] and S[i][j] < current_S / X:
            heapq.heappush(heap, (-S[i][j], i, j))
    
    while heap:
        current_S_neg, i, j = heapq.heappop(heap)
        current_S_val = -current_S_neg
        
        if absorbed[i][j]:
            continue
        if current_S_val >= current_S / X:
            continue
        
        current_S += current_S_val
        absorbed[i][j] = True
        
        # Add new adjacent cells
        for dx, dy in dirs:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W and not absorbed[ni][nj] and (ni, nj) not in adjacent:
                adjacent.add((ni, nj))
        
        # Re-check all adjacent cells and add eligible ones to the heap
        # To avoid modifying the heap while iterating, we collect all candidates first
        candidates = []
        for (ni, nj) in adjacent:
            if not absorbed[ni][nj] and S[ni][nj] < current_S / X:
                candidates.append((-S[ni][nj], ni, nj))
        for item in candidates:
            heapq.heappush(heap, item)
    
    print(current_S)

if __name__ == "__main__":
    main()