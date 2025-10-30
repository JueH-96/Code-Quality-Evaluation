import sys
sys.setrecursionlimit(1000000)

def main():
    data = sys.stdin.read().split()
    H, W, K = map(int, data[:3])
    lines = data[3:]
    # assign an ID to each empty cell
    id_map = [[-1]*W for _ in range(H)]
    M = 0
    for i in range(H):
        row = lines[i]
        for j, c in enumerate(row):
            if c == '.':
                id_map[i][j] = M
                M += 1
    # build adjacency list of the grid graph (only empty cells)
    neighbors = [[] for _ in range(M)]
    for i in range(H):
        for j in range(W):
            u = id_map[i][j]
            if u < 0:
                continue
            if i > 0 and id_map[i-1][j] >= 0:
                neighbors[u].append(id_map[i-1][j])
            if i+1 < H and id_map[i+1][j] >= 0:
                neighbors[u].append(id_map[i+1][j])
            if j > 0 and id_map[i][j-1] >= 0:
                neighbors[u].append(id_map[i][j-1])
            if j+1 < W and id_map[i][j+1] >= 0:
                neighbors[u].append(id_map[i][j+1])
    # visited flags and answer accumulator
    visited = [False]*M
    ans = [0]
    # DFS with default-argument trick to make lookups fast
    def dfs(u, depth,
            visited_list=visited,
            neighbors_list=neighbors,
            K=K,
            ans_list=ans):
        if depth == K:
            ans_list[0] += 1
            return
        for v in neighbors_list[u]:
            if not visited_list[v]:
                visited_list[v] = True
                dfs(v, depth+1)
                visited_list[v] = False

    # start from each empty cell
    for u in range(M):
        visited[u] = True
        dfs(u, 0)
        visited[u] = False

    # print result
    print(ans[0])

if __name__ == '__main__':
    main()