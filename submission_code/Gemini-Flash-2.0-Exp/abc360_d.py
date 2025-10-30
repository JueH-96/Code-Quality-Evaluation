def solve():
    n, t = map(int, input().split())
    s = input()
    x = list(map(int, input().split()))

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (s[i] == '1' and s[j] == '0' and x[i] < x[j] and (x[j] - x[i]) / 2 <= t) or \
               (s[i] == '0' and s[j] == '1' and x[i] > x[j] and (x[i] - x[j]) / 2 <= t):
                count += 1

    print(count)

solve()