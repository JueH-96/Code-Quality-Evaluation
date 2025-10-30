import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    H = int(data[0]); W = int(data[1]); K = int(data[2])
    grid = data[3:]
    # Flatten grid, assign index = i*W + j
    total = H * W
    empty = [False] * total
    for i in range(H):
        row = grid[i]
        base = i * W
        for j, c in enumerate(row):
            if c == '.':
                empty[base + j] = True
    # Precompute neighbors for each pos
    nbrs = [[] for _ in range(total)]
    for i in range(H):
        for j in range(W):
            p = i * W + j
            if not empty[p]:
                continue
            if i > 0 and empty[p - W]:
                nbrs[p].append(p - W)
            if i < H - 1 and empty[p + W]:
                nbrs[p].append(p + W)
            if j > 0 and empty[p - 1]:
                nbrs[p].append(p - 1)
            if j < W - 1 and empty[p + 1]:
                nbrs[p].append(p + 1)

    ans = 0
    # DFS using recursion
    def dfs(pos, visited_mask, depth):
        nonlocal ans
        if depth == K:
            ans += 1
            return
        for q in nbrs[pos]:
            bit = 1 << q
            if not (visited_mask & bit):
                dfs(q, visited_mask | bit, depth + 1)

    # Start from every empty cell
    for start in range(total):
        if not empty[start]:
            continue
        dfs(start, 1 << start, 0)

    print(ans)

if __name__ == "__main__":
    main()