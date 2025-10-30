n, m = map(int, input().split())
a = list(map(int, input().split()))
s = [input().strip() for _ in range(n)]

for i in range(n):
    current_i = sum(a[j] for j in range(m) if s[i][j] == 'o') + (i + 1)
    others = []
    for j in range(n):
        if j != i:
            sum_j = sum(a[j_prev] for j_prev in range(m) if s[j][j_prev] == 'o') + (j + 1)
            others.append(sum_j)
    max_other = max(others)
    if current_i > max_other:
        print(0)
    else:
        x_problems = [a[j] for j in range(m) if s[i][j] == 'x']
        x_problems.sort(reverse=True)
        sum_needed = max_other + 1 - current_i
        current_sum = 0
        k = 0
        for prob in x_problems:
            current_sum += prob
            k += 1
            if current_sum >= sum_needed:
                print(k)
                break