# YOUR CODE HERE
import sys

import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())

    letters = ['s', 'n', 'u', 'k', 'e']

    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),        (0,1),
                  (1,-1),  (1,0), (1,1)]

    for i in range(H):
        for j in range(W):
            if grid[i][j]=='s':
                for dx, dy in directions:
                    positions = [(i+1,j+1)]
                    flag = True
                    for k in range(1,5):
                        ni = i + dx*k
                        nj = j + dy*k
                        if 0 <= ni < H and 0 <= nj < W:
                            if grid[ni][nj] == letters[k]:
                                positions.append((ni+1,nj+1))
                            else:
                                flag=False
                                break
                        else:
                            flag=False
                            break
                    if flag:
                        for pos in positions:
                            print(f"{pos[0]} {pos[1]}")
                        return


threading.Thread(target=main).start()