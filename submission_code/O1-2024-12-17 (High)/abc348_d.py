def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    import heapq

    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip("
") for _ in range(H)]

    # Locate S and T
    sR = sC = -1
    tR = tC = -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                sR, sC = r, c
            elif grid[r][c] == 'T':
                tR, tC = r, c

    # Read medicines
    N = int(input())
    med_map = {}
    for _ in range(N):
        rr, cc, ee = map(int, input().split())
        rr -= 1  # convert to 0-based
        cc -= 1
        med_map[(rr, cc)] = ee

    # If S == T, we can trivially say "Yes"
    if sR == tR and sC == tC:
        print("Yes")
        return

    # First, check reachability ignoring energy (just obstacles).
    # If T is not reachable at all ignoring obstacles, answer "No".
    visited = [[False]*W for _ in range(H)]
    dq = deque()
    visited[sR][sC] = True
    dq.append((sR, sC))
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while dq:
        rr, cc = dq.popleft()
        for dr, dc in directions:
            nr, nc = rr+dr, cc+dc
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[nr][nc] and grid[nr][nc] != '#':
                    visited[nr][nc] = True
                    dq.append((nr,nc))

    if not visited[tR][tC]:
        print("No")
        return

    # Prepare list of "special" cells = {S, T, all medicines that are reachable}
    special_set = set()
    special_set.add((sR, sC))
    special_set.add((tR, tC))
    for (rr, cc), ee in med_map.items():
        if visited[rr][cc]:
            special_set.add((rr, cc))

    # Build the final list of special cells:
    # We'll place S at index 0, T at the last index, everything else in the middle.
    special_set.discard((sR, sC))
    special_set.discard((tR, tC))
    middle_list = list(special_set)  # all reachable medicine cells' coords
    sc = [(sR, sC)] + middle_list + [(tR, tC)]
    sIndex = 0
    tIndex = len(sc) - 1
    M = len(sc)

    # Build a map from (r,c) -> index in sc
    idx_map = {}
    idx_map[(sR, sC)] = 0
    for i, (rr, cc) in enumerate(middle_list, start=1):
        idx_map[(rr, cc)] = i
    idx_map[(tR, tC)] = tIndex

    # eArr[i] = energy of medicine if sc[i] has a medicine, else 0
    eArr = [0]*M
    for i in range(M):
        rr, cc = sc[i]
        if (rr, cc) in med_map:
            eArr[i] = med_map[(rr, cc)]

    # Compute pairwise distances dist[i][j] of special cells ignoring energy,
    # just using BFS in the sub-grid of visited cells (those physically reachable from S).
    dist = [[-1]*M for _ in range(M)]

    def get_dist_from(rstart, cstart):
        dist2 = [[-1]*W for _ in range(H)]
        dist2[rstart][cstart] = 0
        queue = deque()
        queue.append((rstart, cstart))
        while queue:
            rr, cc = queue.popleft()
            dnow = dist2[rr][cc]
            for dr, dc in directions:
                nr, nc = rr+dr, cc+dc
                if 0 <= nr < H and 0 <= nc < W:
                    # We only traverse cells that are reachable ignoring obstacles
                    if visited[nr][nc] and dist2[nr][nc] == -1:
                        dist2[nr][nc] = dnow + 1
                        queue.append((nr, nc))
        return dist2

    # Fill in dist[i][j] by running BFS from each sc[i]
    for i in range(M):
        rS, cS = sc[i]
        dist2 = get_dist_from(rS, cS)
        for j in range(M):
            rT, cT = sc[j]
            dist[i][j] = dist2[rT][cT]

    # Now do a "best-first" search in the small graph of M nodes,
    # storing best[i] = the maximum energy we can have when arriving at node i.
    best_energy = [-1]*M
    # At S, we start with 0 energy, but if there's a medicine, we can use it immediately
    best_energy[sIndex] = eArr[sIndex]  # either 0 or the medicine's energy at S

    # Max-heap of (energy, node). We'll store negative for Python's min-heap
    heap = []
    if best_energy[sIndex] >= 0:
        heapq.heappush(heap, (-best_energy[sIndex], sIndex))

    while heap:
        curr_energy, i = heapq.heappop(heap)
        curr_energy = -curr_energy
        # If this is stale (we found a better energy for node i), skip
        if curr_energy < best_energy[i]:
            continue
        # If we reached T with some nonnegative energy, we are done
        if i == tIndex:
            print("Yes")
            return
        # Try moving to other special nodes j
        for j in range(M):
            d = dist[i][j]
            if d == -1:
                continue  # not physically reachable
            if curr_energy >= d:
                # We can move from i to j
                new_e = curr_energy - d
                # Possibly use medicine at j (only if it increases energy)
                if eArr[j] > new_e:
                    new_e = eArr[j]
                if new_e > best_energy[j]:
                    best_energy[j] = new_e
                    heapq.heappush(heap, (-new_e, j))

    # If we exit the loop without reaching T, it's impossible
    if best_energy[tIndex] >= 0:
        print("Yes")
    else:
        print("No")