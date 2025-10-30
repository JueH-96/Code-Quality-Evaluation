import sys
from collections import deque
from array import array

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    N = H * W
    # Read grid into a flat bytearray: 46=='.', 35=='#'
    grid = bytearray(N)
    for i in range(H):
        line = input().rstrip('
')
        base = i * W
        # assign row
        for j, ch in enumerate(line):
            grid[base + j] = ord(ch)

    # Precompute which empty cells are unblocked:
    # unblocked if grid[id]=='.' and no neighbor is '#'
    is_unblocked = bytearray(N)  # 1 if unblocked, else 0
    for idx in range(N):
        if grid[idx] != 46:  # not '.'
            continue
        i = idx // W
        j = idx - i * W
        # check 4 neighbors for any '#'
        blocked = False
        # up
        if i > 0 and grid[idx - W] == 35:
            blocked = True
        # down
        if not blocked and i + 1 < H and grid[idx + W] == 35:
            blocked = True
        # left
        if not blocked and j > 0 and grid[idx - 1] == 35:
            blocked = True
        # right
        if not blocked and j + 1 < W and grid[idx + 1] == 35:
            blocked = True
        if not blocked:
            is_unblocked[idx] = 1

    # visited flags for unblocked BFS
    visited = bytearray(N)  # 1 if visited in unblocked BFS
    # array to mark blocked-empty cells counted per component
    # we store last comp_id that visited that blocked cell
    blocked_visit = array('i', [-1] * N)

    ans = 1
    comp_id = 0
    # BFS over each unvisited unblocked cell
    for start in range(N):
        if is_unblocked[start] and not visited[start]:
            # new component
            dq = deque()
            visited[start] = 1
            dq.append(start)
            comp_sz = 0
            adj_blocked_cnt = 0
            while dq:
                u = dq.popleft()
                comp_sz += 1
                i = u // W
                j = u - i * W
                # neighbors
                # up
                if i > 0:
                    v = u - W
                    if grid[v] == 46:
                        if is_unblocked[v]:
                            if not visited[v]:
                                visited[v] = 1
                                dq.append(v)
                        else:
                            # blocked-empty neighbor
                            if blocked_visit[v] != comp_id:
                                blocked_visit[v] = comp_id
                                adj_blocked_cnt += 1
                # down
                if i + 1 < H:
                    v = u + W
                    if grid[v] == 46:
                        if is_unblocked[v]:
                            if not visited[v]:
                                visited[v] = 1
                                dq.append(v)
                        else:
                            if blocked_visit[v] != comp_id:
                                blocked_visit[v] = comp_id
                                adj_blocked_cnt += 1
                # left
                if j > 0:
                    v = u - 1
                    if grid[v] == 46:
                        if is_unblocked[v]:
                            if not visited[v]:
                                visited[v] = 1
                                dq.append(v)
                        else:
                            if blocked_visit[v] != comp_id:
                                blocked_visit[v] = comp_id
                                adj_blocked_cnt += 1
                # right
                if j + 1 < W:
                    v = u + 1
                    if grid[v] == 46:
                        if is_unblocked[v]:
                            if not visited[v]:
                                visited[v] = 1
                                dq.append(v)
                        else:
                            if blocked_visit[v] != comp_id:
                                blocked_visit[v] = comp_id
                                adj_blocked_cnt += 1

            # degree of freedom = comp_sz + number of distinct blocked-empty neighbors
            total = comp_sz + adj_blocked_cnt
            if total > ans:
                ans = total
            comp_id += 1

    # blocked-empty cells alone have degree 1,
    # ans was initialized to 1, so covers that case too.
    print(ans)

if __name__ == "__main__":
    main()