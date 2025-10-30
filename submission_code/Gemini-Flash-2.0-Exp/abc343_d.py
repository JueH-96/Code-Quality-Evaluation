def solve():
    n, t = map(int, input().split())
    a = []
    b = []
    for _ in range(t):
        ai, bi = map(int, input().split())
        a.append(ai)
        b.append(bi)

    scores = [0] * n
    for i in range(t):
        scores[a[i]-1] += b[i]
        print(len(set(scores)))

solve()