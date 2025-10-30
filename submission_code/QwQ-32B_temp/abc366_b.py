n = int(input())
strings = [input().strip() for _ in range(n)]
max_len = max(len(s) for s in strings)

for j in range(1, max_len + 1):
    row = []
    for s in reversed(strings):
        if j <= len(s):
            row.append(s[j-1])
        else:
            row.append('*')
    row_str = ''.join(row).rstrip('*')
    print(row_str)