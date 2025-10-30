def solve():
    n, m = map(int, input().split())
    superiority = []
    for _ in range(m):
        superiority.append(list(map(int, input().split())))

    stronger = [[False] * n for _ in range(n)]
    for a, b in superiority:
        stronger[a - 1][b - 1] = True

    changed = True
    while changed:
        changed = False
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if stronger[i][k] and stronger[k][j] and not stronger[i][j]:
                        stronger[i][j] = True
                        changed = True

    strongest_candidates = []
    for p in range(n):
        is_strongest = True
        for q in range(n):
            if p == q:
                continue
            if not stronger[p][q]:
                is_strongest = False
                break
        if is_strongest:
            strongest_candidates.append(p + 1)

    if len(strongest_candidates) == 1:
        print(strongest_candidates[0])
    else:
        print(-1)

# YOUR CODE HERE
solve()