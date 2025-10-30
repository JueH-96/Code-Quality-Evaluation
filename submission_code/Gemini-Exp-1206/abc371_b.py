def solve():
    n, m = map(int, input().split())
    families = [0] * n
    for _ in range(m):
        a, b = input().split()
        a = int(a) - 1
        if b == 'M':
            if families[a] == 0:
                print("Yes")
                families[a] = 1
            else:
                print("No")
        else:
            print("No")

solve()