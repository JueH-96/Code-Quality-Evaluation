# YOUR CODE HERE
from collections import deque
import sys

def solve():
    N = int(sys.stdin.readline())
    grid = []
    players = []
    for i in range(N):
        line = sys.stdin.readline().strip()
        grid.append(line)
        for j, c in enumerate(line):
            if c == 'P':
                players.append((i, j))
                if len(players) == 2:
                    break
    if len(players) !=2:
        print(-1)
        return
    (x1, y1), (x2, y2) = players
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    def sort_positions(p1, p2):
        return p1 + p2 if p1 <= p2 else p2 + p1
    visited = set()
    initial = sort_positions((x1, y1), (x2, y2))
    visited.add(initial)
    queue = deque()
    queue.append((x1, y1, x2, y2))
    steps =0
    while queue:
        size = len(queue)
        for _ in range(size):
            cx1, cy1, cx2, cy2 = queue.popleft()
            if cx1 == cx2 and cy1 == cy2:
                print(steps)
                return
            for dx, dy in directions:
                # Move player 1
                nx1, ny1 = cx1 + dx, cy1 + dy
                if 0 <= nx1 < N and 0 <= ny1 < N and grid[nx1][ny1] != '#':
                    new_p1 = (nx1, ny1)
                else:
                    new_p1 = (cx1, cy1)
                # Move player 2
                nx2, ny2 = cx2 + dx, cy2 + dy
                if 0 <= nx2 < N and 0 <= ny2 < N and grid[nx2][ny2] != '#':
                    new_p2 = (nx2, ny2)
                else:
                    new_p2 = (cx2, cy2)
                # Sort positions to avoid duplicates
                sorted_state = sort_positions(new_p1, new_p2)
                if sorted_state not in visited:
                    if new_p1 == new_p2:
                        print(steps +1)
                        return
                    visited.add(sorted_state)
                    queue.append((sorted_state[0], sorted_state[1], sorted_state[2], sorted_state[3]))
        steps +=1
    print(-1)