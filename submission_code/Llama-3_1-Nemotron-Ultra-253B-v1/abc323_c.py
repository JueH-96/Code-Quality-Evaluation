n, m = map(int, input().split())
a = list(map(int, input().split()))
s = [input().strip() for _ in range(n)]

for i in range(n):
    sum_o = sum(a[j] for j in range(m) if s[i][j] == 'o')
    original_total = sum_o + (i + 1)
    max_other = 0
    for k in range(n):
        if k == i:
            continue
        sum_ok = sum(a[j] for j in range(m) if s[k][j] == 'o')
        total_k = sum_ok + (k + 1)
        if total_k > max_other:
            max_other = total_k
    required = max_other - original_total + 1
    if required <= 0:
        print(0)
    else:
        unsolved = [a[j] for j in range(m) if s[i][j] == 'x']
        unsolved.sort(reverse=True)
        current_sum = 0
        count = 0
        for num in unsolved:
            current_sum += num
            count += 1
            if current_sum >= required:
                print(count)
                break