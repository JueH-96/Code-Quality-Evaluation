def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    count = 0
    for s in range(1, n + 1):
        for t in range(1, n + 1):
            if s == t:
                continue
            distance = 0
            if s < t:
                for i in range(s - 1, t - 1):
                    distance += a[i]
            else:
                for i in range(s - 1, n):
                    distance += a[i]
                for i in range(0, t - 1):
                    distance += a[i]
            if distance % m == 0:
                count += 1
    print(count)

if __name__ == '__main__':
    solve()