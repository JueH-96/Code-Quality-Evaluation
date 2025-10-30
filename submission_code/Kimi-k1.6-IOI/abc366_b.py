n = int(input())
s_list = [input().strip() for _ in range(n)]
cols = s_list[::-1]
m = max(len(s) for s in s_list)
result = []
for j in range(1, m + 1):
    row_chars = []
    for col in cols:
        if len(col) >= j:
            row_chars.append(col[j-1])
        else:
            row_chars.append('*')
    t_j = ''.join(row_chars).rstrip('*')
    result.append(t_j)
print('
'.join(result))