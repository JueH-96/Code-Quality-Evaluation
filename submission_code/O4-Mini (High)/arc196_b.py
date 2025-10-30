import sys
def main():
    input = sys.stdin.readline
    MOD = 998244353

    T = int(input())
    out = []
    for _ in range(T):
        H,W = map(int, input().split())
        # DSU on H+W vertices: rows 0..H-1, columns H..H+W-1
        parent = [-1] * (H+W)
        def find(x):
            # path‐halving
            while parent[x] >= 0:
                if parent[parent[x]] >= 0:
                    parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra = find(a); rb = find(b)
            if ra == rb: return
            # union by size
            if parent[ra] > parent[rb]:
                ra,rb = rb,ra
            parent[ra] += parent[rb]
            parent[rb] = ra

        rowA_par = [0]*H
        colA_par = [0]*W
        row_has_B = [False]*H
        col_has_B = [False]*W

        # read grid, build parity and DSU
        for i in range(H):
            s = input().strip()
            for j,ch in enumerate(s):
                if ch == 'A':
                    rowA_par[i] ^= 1
                    colA_par[j] ^= 1
                else:
                    row_has_B[i] = True
                    col_has_B[j] = True
                    union(i, H+j)

        # parity check
        if any(rowA_par):
            out.append('0')
            continue
        if any(colA_par):
            out.append('0')
            continue

        # R0 = # all‐A rows, C0 = # all‐A cols
        R0 = row_has_B.count(False)
        C0 = col_has_B.count(False)

        # count bipartite‐connected components among those with at least one B
        visited = bytearray(H+W)
        J = 0
        for i in range(H):
            if row_has_B[i]:
                r = find(i)
                if not visited[r]:
                    visited[r] = 1
                    J += 1
        for j in range(W):
            if col_has_B[j]:
                r = find(H+j)
                if not visited[r]:
                    visited[r] = 1
                    J += 1

        exp = R0 + C0 + J
        out.append(str(pow(2, exp, MOD)))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()