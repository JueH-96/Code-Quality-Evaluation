def solve():
    n = int(input())
    s = list(map(int, input().split()))
    s.sort()
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = s[i]
            b = s[j]
            c = 2 * b - a
            if c in s and c > b:
                count += 1
    print(count)

solve()