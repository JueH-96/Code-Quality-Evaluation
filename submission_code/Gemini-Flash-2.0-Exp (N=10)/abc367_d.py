def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    count = 0
    for s in range(n):
        for t in range(n):
            if s != t:
                dist = 0
                curr = s
                while curr != t:
                    dist += a[curr]
                    curr = (curr + 1) % n
                if dist % m == 0:
                    count += 1
    print(count)

solve()