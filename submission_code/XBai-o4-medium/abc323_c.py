n, m = map(int, input().split())
a = list(map(int, input().split()))
s_list = [input().strip() for _ in range(n)]

original_totals = []
for i in range(n):
    total = i + 1
    s = s_list[i]
    for j in range(m):
        if s[j] == 'o':
            total += a[j]
    original_totals.append(total)

for i in range(n):
    max_other = 0
    for j in range(n):
        if i != j:
            if original_totals[j] > max_other:
                max_other = original_totals[j]
    current = original_totals[i]
    required_diff = max(0, max_other - current + 1)
    if required_diff == 0:
        print(0)
        continue
    unsolved = []
    s_i = s_list[i]
    for j in range(m):
        if s_i[j] == 'x':
            unsolved.append(a[j])
    unsolved.sort(reverse=True)
    sum_so_far = 0
    count = 0
    for score in unsolved:
        sum_so_far += score
        count += 1
        if sum_so_far >= required_diff:
            break
    print(count)