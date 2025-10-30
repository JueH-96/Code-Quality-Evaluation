n, m = map(int, input().split())
a = list(map(int, input().split()))
s_list = [input().strip() for _ in range(n)]

current_totals = []
for idx in range(n):
    s = s_list[idx]
    total = sum(a[j] for j in range(m) if s[j] == 'o')
    total += idx + 1
    current_totals.append(total)

for i in range(n):
    max_other = max(current_totals[k] for k in range(n) if k != i)
    current_i_total = current_totals[i]
    if current_i_total > max_other:
        print(0)
        continue
    required = max_other - current_i_total + 1
    unsolved = [a[j] for j in range(m) if s_list[i][j] == 'x']
    unsolved_sorted = sorted(unsolved, reverse=True)
    sum_added = 0
    count = 0
    for val in unsolved_sorted:
        sum_added += val
        count += 1
        if sum_added >= required:
            break
    print(count)