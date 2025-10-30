def solve():
    n, m, a, b = map(int, input().split())
    bad_ranges = []
    for _ in range(m):
        bad_ranges.append(tuple(map(int, input().split())))

    bad_squares = set()
    for l, r in bad_ranges:
        for i in range(l, r + 1):
            bad_squares.add(i)

    q = [1]
    visited = {1}

    while q:
        curr = q.pop(0)
        if curr == n:
            print("Yes")
            return

        for i in range(a, b + 1):
            next_square = curr + i
            if next_square <= n and next_square not in bad_squares and next_square not in visited:
                q.append(next_square)
                visited.add(next_square)

    print("No")

solve()