t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    s = input().strip()
    b = [i for i in range(n) if s[i] == 'B']
    if not b:
        print(0)
        continue
    b = [b[0] - i + (i != 0) for i in range(len(b))]
    b.sort()
    ans = 1
    for i in range(1, len(b)):
        if b[i] - b[i - 1] > k:
            ans += 1
    print(ans)