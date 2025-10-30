# YOUR CODE HERE
import sys

def main():
    H_W = sys.stdin.readline().strip()
    while H_W == '':
        H_W = sys.stdin.readline().strip()
    H, W = map(int, H_W.split())
    grid = []
    for _ in range(H):
        row = sys.stdin.readline().strip()
        while len(row) < W:
            row += sys.stdin.readline().strip()
        grid.append(row)
    
    target = ['s', 'n', 'u', 'k', 'e']
    directions = [(-1, -1), (-1, 0), (-1,1), (0, -1), (0,1), (1,-1), (1,0), (1,1)]
    found = False
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    cells = []
                    for k in range(5):
                        ni = i + k*dx
                        nj = j + k*dy
                        if 0 <= ni < H and 0 <= nj < W:
                            if grid[ni][nj] == target[k]:
                                cells.append( (ni+1, nj+1) )
                            else:
                                break
                        else:
                            break
                    if len(cells) == 5:
                        for cell in cells:
                            print(cell[0], cell[1])
                        return

if __name__ == "__main__":
    main()