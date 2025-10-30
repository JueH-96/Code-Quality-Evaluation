n = int(input())
D = list(map(int, input().split()))
count = 0

for i in range(1, n + 1):
    s_i = str(i)
    if not all(c == s_i[0] for c in s_i):
        continue
    max_j = D[i - 1]
    for j in range(1, max_j + 1):
        s_j = str(j)
        if not all(c == s_j[0] for c in s_j):
            continue
        if s_i[0] == s_j[0]:
            count += 1

print(count)