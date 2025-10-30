t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    count = 0
    last = -1
    for i in range(n):
        if s[i] == 'B' and i > last:
            x = min(i, n - k)
            count += 1
            last = x + k - 1
    print(count)