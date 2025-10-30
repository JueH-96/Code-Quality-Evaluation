import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W, N = map(int, data[0].split())
    T = data[1].strip()
    grid = []
    for i in range(2, 2+H):
        grid.append(data[i].strip())
    
    revT = T[::-1]
    
    cur = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                cur.add((i, j))
                
    for c in revT:
        nxt = set()
        for (i, j) in cur:
            if c == 'L':
                ni, nj = i, j + 1
            elif c == 'R':
                ni, nj = i, j - 1
            elif c == 'U':
                ni, nj = i + 1, j
            elif c == 'D':
                ni, nj = i - 1, j
            else:
                continue
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                nxt.add((ni, nj))
        cur = nxt
        
    print(len(cur))

if __name__ == "__main__":
    main()