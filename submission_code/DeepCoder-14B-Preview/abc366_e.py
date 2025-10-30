n = int(input())
s_list = [input().strip() for _ in range(n)]
m = max(len(s) for s in s_list) if s_list else 0

t_list = []

for j in range(1, m + 1):
    temp = []
    for i in range(n):
        if j <= len(s_list[i]):
            temp.append(s_list[i][j-1])
        else:
            temp.append('*')
    reversed_temp = temp[::-1]
    t_j = ''.join(reversed_temp).rstrip('*')
    t_list.append(t_j)

for t in t_list:
    print(t)