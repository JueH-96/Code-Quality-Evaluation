n = int(input())
strings = [input().strip() for _ in range(n)]
m = max(len(s) for s in strings)

for j in range(1, m + 1):
    column = []
    for r in range(1, n + 1):
        i = n - r + 1
        s_r = strings[i - 1]
        if j <= len(s_r):
            column.append(s_r[j - 1])
        else:
            column.append('*')
    temp_str = ''.join(column).rstrip('*')
    print(temp_str)