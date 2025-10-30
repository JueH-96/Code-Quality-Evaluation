def solve():
    n, q = map(int, input().split())
    s = input()
    for _ in range(q):
        l, r = map(int, input().split())
        count = 0
        for p in range(l, r):
            if s[p - 1] == s[p]:
                count += 1
        print(count)

solve()