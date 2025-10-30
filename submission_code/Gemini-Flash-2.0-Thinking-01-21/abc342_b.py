def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        a, b = map(int, input().split())
        index_a = -1
        index_b = -1
        for i in range(n):
            if p[i] == a:
                index_a = i
            if p[i] == b:
                index_b = i

        if index_a < index_b:
            print(a)
        else:
            print(b)

solve()