def solve():
    n, q = map(int, input().split())
    queries = [int(input()) for _ in range(q)]

    possible_fixed = {0, n * n}

    for k in queries:
        if k in possible_fixed:
            print("Yes")
        else:
            print("No")

solve()