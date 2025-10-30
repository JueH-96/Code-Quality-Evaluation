n = int(input())
strings = [input().strip() for _ in range(n)]
max_len = max(len(s) for s in strings)

for j in range(1, max_len + 1):
    row = []
    for k in range(1, n + 1):
        i = n - k + 1
        s = strings[i - 1]
        if j <= len(s):
            c = s[j - 1]
        else:
            c = '*'
        row.append(c)
    # Trim trailing stars
    last = len(row) - 1
    while last >= 0 and row[last] == '*':
        last -= 1
    print(''.join(row[:last + 1]))