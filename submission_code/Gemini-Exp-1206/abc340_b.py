def solve():
    q = int(input())
    a = []
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            a.append(query[1])
        else:
            print(a[-query[1]])

solve()