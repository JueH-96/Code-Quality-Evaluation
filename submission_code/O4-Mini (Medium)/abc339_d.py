import sys
import threading
from collections import deque

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    grid = data[1:]
    M = N * N

    # find the two player positions and mark obstacles
    pos_list = []
    blocked = [False] * M
    for i in range(N):
        row = grid[i]
        for j in range(N):
            ch = row[j]
            idx = i * N + j
            if ch == '#':
                blocked[idx] = True
            elif ch == 'P':
                pos_list.append(idx)
    # initial positions
    p0, p1 = pos_list
    if p0 == p1:
        print(0)
        return

    # we will represent a state by (a,b) with a<=b
    a0, b0 = (p0, p1) if p0 <= p1 else (p1, p0)

    # directions: up, down, left, right
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]

    # precompute neighbor moves for each cell & each direction
    neighbor = [None] * M
    for pos in range(M):
        r = pos // N
        c = pos % N
        lst = [0]*4
        for d, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                npos = nr * N + nc
                if not blocked[npos]:
                    lst[d] = npos
                else:
                    lst[d] = pos
            else:
                lst[d] = pos
        neighbor[pos] = lst

    # for visited, we only store states (a,b) with a<=b
    # pack (a,b) into one int: code = (a<<SHIFT)|b
    # and keep a visited bitset per 'a', bit b
    SHIFT = (M-1).bit_length()
    maskB = (1 << SHIFT) - 1

    visited = [0] * M
    def mark(a, b):
        visited[a] |= (1 << b)
    def is_vis(a, b):
        return (visited[a] >> b) & 1

    dq = deque()
    mark(a0, b0)
    dq.append((a0 << SHIFT) | b0)
    depth = 0

    while dq:
        # process current BFS layer
        for _ in range(len(dq)):
            code = dq.popleft()
            a = code >> SHIFT
            b = code & maskB
            # try all 4 moves
            na_list = neighbor[a]
            nb_list = neighbor[b]
            for d in range(4):
                na = na_list[d]
                nb = nb_list[d]
                # if they meet
                if na == nb:
                    print(depth + 1)
                    return
                # sort to maintain a<=b
                if na > nb:
                    na, nb = nb, na
                # not visited?
                if not is_vis(na, nb):
                    mark(na, nb)
                    dq.append((na << SHIFT) | nb)
        depth += 1

    # if BFS finishes with no meet
    print(-1)

if __name__ == "__main__":
    main()