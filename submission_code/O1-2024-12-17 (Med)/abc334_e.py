def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    S_list = input_data[2:]  # Each element is a row string like ".#.#"

    # We work modulo this prime:
    MOD = 998244353

    # A helper to compute modular inverse (since MOD is prime we can use Fermat's little theorem):
    def modinv(x, m=MOD):
        # x^(m-2) mod m
        return pow(x, m-2, m)

    # We label connected components of the green cells ('#') using a BFS or DFS.
    from collections import deque

    comp = [-1] * (H*W)  # comp[id] will store the component index of the cell (if it's green)
    def get_id(r, c):
        return r*W + c

    drdc = [(1,0),(0,1),(-1,0),(0,-1)]

    comp_index = 0
    for i in range(H):
        for j in range(W):
            if S_list[i][j] == '#' and comp[get_id(i,j)] == -1:
                # BFS to mark all cells in this connected component
                queue = deque()
                queue.append((i,j))
                comp[get_id(i,j)] = comp_index
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in drdc:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if S_list[nr][nc] == '#' and comp[get_id(nr,nc)] == -1:
                                comp[get_id(nr,nc)] = comp_index
                                queue.append((nr,nc))
                comp_index += 1

    oldCC = comp_index  # number of green connected components initially

    # Now compute the sum of the new number of green components after painting each red cell green.
    sum_f = 0
    M = 0  # number of red cells
    for i in range(H):
        for j in range(W):
            if S_list[i][j] == '.':  # red cell
                M += 1
                neighbors_set = set()
                for dr, dc in drdc:
                    nr, nc = i+dr, j+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if S_list[nr][nc] == '#':
                            neighbors_set.add(comp[get_id(nr,nc)])
                k = len(neighbors_set)
                # new #components = oldCC + 1 - k
                f = oldCC + 1 - k
                sum_f = (sum_f + f) % MOD

    # Expected value = (sum_f / M) mod 998244353 = sum_f * inv(M) mod
    inv_M = modinv(M)
    ans = (sum_f * inv_M) % MOD
    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()