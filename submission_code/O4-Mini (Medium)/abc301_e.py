import sys
import threading
def main():
    import sys
    from collections import deque
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it)); W = int(next(it)); T = int(next(it))
    grid = []
    for _ in range(H):
        grid.append(list(next(it).strip()))
    # find positions
    sx = sy = gx = gy = -1
    candies = []
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == 'S':
                sx, sy = i, j
            elif c == 'G':
                gx, gy = i, j
            elif c == 'o':
                candies.append((i, j))
    Nc = len(candies)
    # build positions: index 0 = S, 1..Nc = candies, Nc+1 = G
    positions = [(sx, sy)] + candies + [(gx, gy)]
    N = Nc + 2
    # compute pairwise distances
    dist_mat = [[-1]*N for _ in range(N)]
    # BFS from each special
    Hm = H; Wm = W
    for idx in range(N):
        sx0, sy0 = positions[idx]
        dist2d = [[-1]*Wm for _ in range(Hm)]
        dq = deque()
        dist2d[sx0][sy0] = 0
        dq.append((sx0, sy0))
        while dq:
            x, y = dq.popleft()
            d0 = dist2d[x][y] + 1
            # four directions
            nx = x-1; ny = y
            if nx >= 0 and grid[nx][ny] != '#' and dist2d[nx][ny] < 0:
                dist2d[nx][ny] = d0; dq.append((nx, ny))
            nx = x+1; ny = y
            if nx < Hm and grid[nx][ny] != '#' and dist2d[nx][ny] < 0:
                dist2d[nx][ny] = d0; dq.append((nx, ny))
            nx = x; ny = y-1
            if ny >= 0 and grid[nx][ny] != '#' and dist2d[nx][ny] < 0:
                dist2d[nx][ny] = d0; dq.append((nx, ny))
            nx = x; ny = y+1
            if ny < Wm and grid[nx][ny] != '#' and dist2d[nx][ny] < 0:
                dist2d[nx][ny] = d0; dq.append((nx, ny))
        # fill in dist_mat[idx][*]
        for jdx in range(N):
            x2, y2 = positions[jdx]
            dist_mat[idx][jdx] = dist2d[x2][y2]
    # shorthand
    INF = T + 1
    # if no candies
    if Nc == 0:
        d0 = dist_mat[0][1]
        if d0 != -1 and d0 <= T:
            print(0)
        else:
            print(-1)
        return
    # Precompute bitcounts
    maxmask = 1 << Nc
    bitcount = [0] * maxmask
    for m in range(1, maxmask):
        bitcount[m] = bitcount[m >> 1] + (m & 1)
    # dp[mask][i] flattened: size maxmask * Nc
    # use array for compact storage
    from array import array
    dp = array('I', [INF]) * (maxmask * Nc)
    # init
    for i in range(Nc):
        d0 = dist_mat[0][i+1]
        if d0 != -1 and d0 <= T:
            dp[(1<<i)*Nc + i] = d0
    # DP transitions
    for mask in range(maxmask):
        base = mask * Nc
        # for each last candy i in mask
        for i in range(Nc):
            if not (mask & (1<<i)):
                continue
            cur = dp[base + i]
            if cur > T:
                continue
            # try to go to a new candy j
            # from candy i (idx i+1 in dist_mat)
            di = dist_mat[i+1]
            # j is candy index 0..Nc-1 representing dist_mat idx j+1
            for j in range(Nc):
                if mask & (1<<j):
                    continue
                d_ij = di[j+1]
                if d_ij < 0:
                    continue
                nd = cur + d_ij
                if nd > T:
                    continue
                nm = mask | (1<<j)
                pos = nm * Nc + j
                if nd < dp[pos]:
                    dp[pos] = nd
    # compute answer
    ans = -1
    # direct path without candies
    d_sg = dist_mat[0][Nc+1]
    if d_sg != -1 and d_sg <= T:
        ans = 0
    # try masks with candies
    for mask in range(1, maxmask):
        bc = bitcount[mask]
        # only if bc > current ans (prune)
        if bc <= ans:
            continue
        base = mask * Nc
        for i in range(Nc):
            if not (mask & (1<<i)):
                continue
            cur = dp[base + i]
            if cur > T:
                continue
            # from candy i to goal
            d_ig = dist_mat[i+1][Nc+1]
            if d_ig < 0:
                continue
            if cur + d_ig <= T:
                # valid
                if bc > ans:
                    ans = bc
                break
    print(ans)

if __name__ == "__main__":
    main()