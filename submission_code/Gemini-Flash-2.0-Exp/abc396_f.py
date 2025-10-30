def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    for k in range(m):
        b = [(x + k) % m for x in a]
        inversion_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if b[i] > b[j]:
                    inversion_count += 1
        print(inversion_count)

solve()