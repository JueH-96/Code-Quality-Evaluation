n, m = map(int, input().split())
a = list(map(int, input().split()))
s = [input().strip() for _ in range(n)]

sums = []
for si in s:
    total = 0
    for j in range(m):
        if si[j] == 'o':
            total += a[j]
    sums.append(total)

current_scores = [sums[i] + (i + 1) for i in range(n)]

for i in range(n):
    max_other = 0
    for j in range(n):
        if j != i and current_scores[j] > max_other:
            max_other = current_scores[j]
    required = max(0, max_other - current_scores[i] + 1)
    if required == 0:
        print(0)
    else:
        unsolved = []
        for j in range(m):
            if s[i][j] == 'x':
                unsolved.append(a[j])
        unsolved.sort(reverse=True)
        sum_so_far = 0
        count = 0
        for num in unsolved:
            sum_so_far += num
            count += 1
            if sum_so_far >= required:
                break
        print(count)