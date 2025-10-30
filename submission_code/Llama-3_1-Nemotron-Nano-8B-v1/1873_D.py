import bisect

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    B = [i for i, c in enumerate(s) if c == 'B']
    m = len(B)
    if m == 0:
        print(0)
        continue
    count = 0
    i = 0
    while i < m:
        current = B[i]
        s_start = min(current, n - k)
        end = s_start + k - 1
        j = bisect.bisect_right(B, end, i, m) - 1
        count += 1
        i = j + 1
    print(count)