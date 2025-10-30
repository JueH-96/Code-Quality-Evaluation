def solve():
    N, A, B = map(int, input().split())
    D = list(map(int, input().split()))

    total_days = A + B
    max_days = max(D)

    if max_days > total_days:
        print("No")
    else:
        print("Yes")

solve()