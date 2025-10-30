import sys

def main():
    import sys
    input = sys.stdin.readline

    H, W, M = map(int, input().split())
    ops = [None] * M
    for i in range(M):
        t, a, x = map(int, input().split())
        ops[i] = (t, a, x)

    painted_rows = [False] * (H + 1)
    painted_cols = [False] * (W + 1)
    remain_rows = H
    remain_cols = W

    from collections import defaultdict
    ans = defaultdict(int)

    # Process operations in reverse to count "newly painted" cells
    for t, a, x in reversed(ops):
        if t == 1:
            # paint row a
            if not painted_rows[a]:
                painted_rows[a] = True
                ans[x] += remain_cols
                remain_rows -= 1
        else:
            # paint column a
            if not painted_cols[a]:
                painted_cols[a] = True
                ans[x] += remain_rows
                remain_cols -= 1

    # any cells never touched by any row or column op remain color 0
    if remain_rows > 0 and remain_cols > 0:
        ans[0] += remain_rows * remain_cols

    # collect and sort colors with positive counts
    items = [(c, cnt) for c, cnt in ans.items() if cnt > 0]
    items.sort()  # sort by color ascending

    # output
    print(len(items))
    for c, cnt in items:
        print(c, cnt)

if __name__ == "__main__":
    main()