def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    if k == 2:
        for x in a:
            if x % 2 == 0:
                print(0)
                return
        print(1)
    elif k == 3:
        for x in a:
            if x % 3 == 0:
                print(0)
                return
        ans = float('inf')
        for x in a:
            ans = min(ans, (3 - x % 3) % 3)
        print(ans)
    elif k == 4:
        for x in a:
            if x % 4 == 0:
                print(0)
                return
        even_count = 0
        for x in a:
            if x % 2 == 0:
                even_count += 1
        if even_count >= 2:
            print(0)
        elif even_count == 1:
            print(1)
        else:
            print(2)
    elif k == 5:
        for x in a:
            if x % 5 == 0:
                print(0)
                return
        ans = float('inf')
        for x in a:
            ans = min(ans, (5 - x % 5) % 5)
        print(ans)

t = int(input())
for _ in range(t):
    solve()