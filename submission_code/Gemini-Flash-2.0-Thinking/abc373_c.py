def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    max_a = -float('inf')
    for x in a:
        max_a = max(max_a, x)

    max_b = -float('inf')
    for y in b:
        max_b = max(max_b, y)

    print(max_a + max_b)

solve()