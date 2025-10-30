def solve():
    K, Sx, Sy, Tx, Ty = map(int, input().split())

    def get_tile(x, y):
        i = x // K
        j = y // K
        if (i % 2 == j % 2):
            k = y % K
        else:
            k = x % K
        return i, j, k

    si, sj, sk = get_tile(Sx, Sy)
    ti, tj, tk = get_tile(Tx, Ty)

    if si == ti and sj == tj:
        print(0)
        return

    q = [(si, sj, sk, 0)]
    visited = set([(si, sj, sk)])

    while q:
        i, j, k, dist = q.pop(0)

        if i == ti and j == tj:
            print(dist)
            return

        neighbors = []
        if (i % 2 == j % 2):
            for nk in range(K):
                neighbors.append((i, j, nk))
            if i > -10**18 // K:
                neighbors.append((i - 1, j, (i - 1) % 2 != j % 2 and (Sx if i*K <= Sx < (i+1)*K else i*K) % K or 0))
            if i < 10**18 // K:
                neighbors.append((i + 1, j, (i + 1) % 2 != j % 2 and (Sx if (i+1)*K <= Sx < (i+2)*K else (i+1)*K) % K or 0))
            if j > -10**18 // K:
                neighbors.append((i, j - 1, i % 2 != (j-1) % 2 and (Sx if i*K <= Sx < (i+1)*K else i*K) % K or 0))
            if j < 10**18 // K:
                neighbors.append((i, j + 1, i % 2 != (j+1) % 2 and (Sx if i*K <= Sx < (i+1)*K else i*K) % K or 0))
        else:
            for nk in range(K):
                neighbors.append((i, j, nk))
            if i > -10**18 // K:
                neighbors.append((i - 1, j, (i - 1) % 2 != j % 2 and (Sx if (i-1)*K <= Sx < i*K else (i-1)*K) % K or 0))
            if i < 10**18 // K:
                neighbors.append((i + 1, j, (i + 1) % 2 != j % 2 and (Sx if i*K <= Sx < (i+1)*K else i*K) % K or 0))
            if j > -10**18 // K:
                neighbors.append((i, j - 1, i % 2 != (j-1) % 2 and (Sx if i*K <= Sx < (i+1)*K else i*K) % K or 0))
            if j < 10**18 // K:
                neighbors.append((i, j + 1, i % 2 != (j+1) % 2 and (Sx if i*K <= Sx < (i+1)*K else i*K) % K or 0))
        

        for ni, nj, nk in neighbors:
            if (ni, nj, nk) not in visited:
                visited.add((ni, nj, nk))
                q.append((ni, nj, nk, dist + 1))

T = int(input())
for _ in range(T):
    solve()