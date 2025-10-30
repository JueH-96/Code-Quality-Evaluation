def solve():
    b = int(input())
    for a in range(1, int(b**0.5) + 2):
        if a**a == b:
            print(a)
            return
    print(-1)

solve()