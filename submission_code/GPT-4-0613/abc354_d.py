def solve():
    a, b, c, d = map(int, input().split())
    a, c = (a + 1) // 2, c // 2
    b, d = (b + 1) // 2, d // 2
    print((c - a + d - b) % 2)

solve()