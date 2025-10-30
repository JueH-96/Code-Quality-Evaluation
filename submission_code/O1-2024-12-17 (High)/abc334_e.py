def main():
    import sys
    from collections import deque

    sys.setrecursionlimit(10**7)
    MOD = 998244353

    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    S = data[2:]  # S[i] is the i-th row string

    # comp[i][j] = -1 if cell (i,j) is red, or not yet visited if green
    # otherwise, it stores the connected component ID of that green cell
    comp = [[-1]*W for _ in range(H)]

    # Directions for 4-neighborhood
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    # Step 1: Find the number of green connected components with BFS
    comp_id = 0
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#' and comp[i][j] == -1:
                # BFS to mark all connected green cells with comp_id
                queue = deque()
                queue.append((i, j))
                comp[i][j] = comp_id
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if S[nr][nc] == '#' and comp[nr][nc] == -1:
                                comp[nr][nc] = comp_id
                                queue.append((nr, nc))
                comp_id += 1
    # comp_id is now the total count of green connected components C
    C = comp_id

    # Step 2: For each red cell, count how many distinct green-component neighbors it has
    M = 0  # number of red cells
    sum_k = 0  # sum of distinct green-component neighbors over all red cells
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.':
                M += 1
                neighbors = set()
                for dr, dc in directions:
                    nr, nc = i + dr, j + dc
                    if 0 <= nr < H and 0 <= nc < W and comp[nr][nc] != -1:
                        neighbors.add(comp[nr][nc])
                sum_k += len(neighbors)

    # Step 3: Expected value E = ( (C+1)*M - sum_k ) / M  (mod 998244353)
    # Numerator
    numerator = ((C + 1) * M - sum_k) % MOD
    # Denominator is M, so we need the modular inverse of M
    inv_M = pow(M, MOD - 2, MOD)  # Fermat's little theorem
    ans = (numerator * inv_M) % MOD

    print(ans)

# Do not forget to call main!
if __name__ == "__main__":
    main()