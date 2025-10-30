def solve():
    n = int(input())
    a = list(map(int, input().split()))

    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1

    candidates = []
    for i in range(n):
        if counts[a[i]] == 1:
            candidates.append((a[i], i + 1))

    if not candidates:
        print("-1")
    else:
        max_val = -1
        max_index = -1
        for val, index in candidates:
            if val > max_val:
                max_val = val
                max_index = index
        print(max_index)

solve()