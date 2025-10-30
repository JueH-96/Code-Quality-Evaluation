# YOUR CODE HERE
from collections import deque

def main():
    import sys
    import threading
    def solve():
        N = int(sys.stdin.readline())
        grid = []
        player_positions = []
        for i in range(N):
            s = sys.stdin.readline().strip()
            grid.append(s)
            for j in range(N):
                if s[j] == 'P':
                    player_positions.append((i, j))
        if len(player_positions) != 2:
            print(-1)
            return
        # Use zero-based indexing
        r1, c1 = player_positions[0]
        r2, c2 = player_positions[1]
        from collections import deque
        queue = deque()
        # State: (r1, c1, r2, c2, steps)
        queue.append((r1, c1, r2, c2, 0))
        visited = set()
        visited.add((r1, c1, r2, c2))
        directions = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
        N = len(grid)
        def is_empty(r, c):
            return 0 <= r < N and 0 <= c < N and grid[r][c] != '#'
        while queue:
            cr1, cc1, cr2, cc2, steps = queue.popleft()
            if (cr1, cc1) == (cr2, cc2):
                print(steps)
                return
            for dr, dc in directions:
                nr1, nc1 = cr1 + dr, cc1 + dc
                nr2, nc2 = cr2 + dr, cc2 + dc
                # For player 1
                if is_empty(nr1, nc1):
                    pr1, pc1 = nr1, nc1
                else:
                    pr1, pc1 = cr1, cc1
                # For player 2
                if is_empty(nr2, nc2):
                    pr2, pc2 = nr2, nc2
                else:
                    pr2, pc2 = cr2, cc2
                if (pr1, pc1) == (pr2, pc2):
                    print(steps+1)
                    return
                if (pr1, pc1, pr2, pc2) not in visited:
                    visited.add((pr1, pc1, pr2, pc2))
                    queue.append((pr1, pc1, pr2, pc2, steps+1))
        print(-1)
    threading.Thread(target=solve).start()
if __name__ == '__main__':
    main()