n, m = map(int, input().split())
a = list(map(int, input().split()))
s = [input().strip() for _ in range(n)]

current_total = []
for j in range(n):
    total = 0
    for i in range(m):
        if s[j][i] == 'o':
            total += a[i]
    total += (j + 1)
    current_total.append(total)

for j in range(n):
    available = []
    for i in range(m):
        if s[j][i] == 'x':
            available.append(a[i])
    available.sort(reverse=True)
    
    max_other = 0
    for k in range(n):
        if k != j:
            if current_total[k] > max_other:
                max_other = current_total[k]
    
    if current_total[j] > max_other:
        print(0)
    else:
        needed = max_other - current_total[j] + 1
        sum_add = 0
        count = 0
        for score in available:
            sum_add += score
            count += 1
            if sum_add >= needed:
                break
        print(count)