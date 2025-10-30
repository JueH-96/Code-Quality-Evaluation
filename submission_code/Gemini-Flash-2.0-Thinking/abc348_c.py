def solve():
    n = int(input())
    beans_by_color = {}
    for _ in range(n):
        a, c = map(int, input().split())
        if c not in beans_by_color:
            beans_by_color[c] = []
        beans_by_color[c].append(a)

    max_min_deliciousness = 0
    for color in beans_by_color:
        min_deliciousness = min(beans_by_color[color])
        max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

    print(max_min_deliciousness)

solve()