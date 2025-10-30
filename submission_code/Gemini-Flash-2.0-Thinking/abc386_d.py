def solve():
    n, m = map(int, input().split())
    colored = []
    for _ in range(m):
        x, y, c = input().split()
        colored.append((int(x), int(y), c))

    min_rt = [0] * n
    max_rt = [n] * n
    min_ct = [0] * n
    max_ct = [n] * n

    for x, y, c in colored:
        r, col = x - 1, y - 1
        if c == 'B':
            min_rt[r] = max(min_rt[r], col + 1)
            min_ct[col] = max(min_ct[col], r + 1)
        else:
            max_rt[r] = min(max_rt[r], col)
            max_ct[col] = min(max_ct[col], r)

    for r in range(n):
        if min_rt[r] > max_rt[r]:
            print("No")
            return
    for c in range(n):
        if min_ct[c] > max_ct[c]:
            print("No")
            return

    for x, y, c in colored:
        r, col = x - 1, y - 1
        if c == 'B':
            if max_rt[r] < col + 1 or max_ct[col] < r + 1:
                print("No")
                return
        else:
            if min_rt[r] >= col + 1 and min_ct[col] >= r + 1:
                print("No")
                return

    print("Yes")

solve()