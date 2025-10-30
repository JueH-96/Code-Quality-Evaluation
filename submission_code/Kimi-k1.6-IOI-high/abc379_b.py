n, k = map(int, input().split())
s = input().strip()

intervals = []
for i in range(n - k + 1):
    if all(c == 'O' for c in s[i:i+k]):
        intervals.append((i, i + k - 1))

intervals.sort(key=lambda x: x[1])

count = 0
last_end = -1
for start, end in intervals:
    if start > last_end:
        count += 1
        last_end = end

print(count)