N = int(input())
for r in range(1, N+1):
    row = []
    for c in range(1, N+1):
        max_rc = max(r, c)
        term = N + 1 - max_rc
        min_rc = min(r, c)
        K = min(min_rc, term)
        if K % 2 == 1:
            row.append('#')
        else:
            row.append('.')
    print(''.join(row))