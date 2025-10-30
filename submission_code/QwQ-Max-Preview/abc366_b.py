n = int(input())
s_list = [input().strip() for _ in range(n)]
m = max(len(s) for s in s_list)

for j in range(1, m + 1):
    row = []
    for k in range(1, n + 1):
        current_i = n - k + 1
        current_s = s_list[current_i - 1]
        if j <= len(current_s):
            c = current_s[j-1]
        else:
            c = '*'
        row.append(c)
    t = ''.join(row).rstrip('*')
    print(t)