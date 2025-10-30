import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    grid = []
    s = None
    t = None
    for i in range(H):
        line = data[idx]
        idx += 1
        grid.append(line)
        for j in range(W):
            if line[j] == 'S':
                s = (i, j)
            elif line[j] == 'T':
                t = (i, j)
    
    medicines = []
    max_e = [[0 for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == '#':
                continue
            elif c == 'S' or c == 'T':
                continue
            elif c == '.':
                continue
            else:
                e = int(c)
                medicines.append((i, j, e))
                if e > max_e[i][j]:
                    max_e[i][j] = e
    
    heap = []
    heapq.heappush(heap, (0, s[0], s[1]))
    
    found = False
    
    while heap:
        current_neg_e, i, j = heapq.heappop(heap)
        current_e = -current_neg_e
        
        if (i, j) == t:
            found = True
            break
        
        if current_e < max_e[i][j]:
            continue
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    continue
                new_e = current_e - 1
                if new_e >= 0 and new_e > max_e[ni][nj]:
                    max_e[ni][nj] = new_e
                    heapq.heappush(heap, (-new_e, ni, nj))
        
        if current_e < max_e[i][j]:
            new_e = max_e[i][j]
            heapq.heappush(heap, (-new_e, i, j))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()