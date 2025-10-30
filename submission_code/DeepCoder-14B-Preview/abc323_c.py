n, m = map(int, input().split())
A = list(map(int, input().split()))
players = [input().strip() for _ in range(n)]

for i in range(n):
    sum_i = 0
    for j in range(m):
        if players[i][j] == 'o':
            sum_i += A[j]
    
    max_other = 0
    for j in range(n):
        if j == i:
            continue
        sum_j = 0
        for k in range(m):
            if players[j][k] == 'o':
                sum_j += A[k]
        total_j = sum_j + (j + 1)
        if total_j > max_other:
            max_other = total_j
    
    D = max_other - (sum_i + (i + 1))
    if D < 0:
        print(0)
        continue
    
    sum_new_min = ((D // 100) + 1) * 100
    unsolved = []
    for j in range(m):
        if players[i][j] == 'x':
            unsolved.append(A[j])
    
    unsolved_sorted = sorted(unsolved, reverse=True)
    prefix = [0]
    current = 0
    for a in unsolved_sorted:
        current += a
        prefix.append(current)
    
    min_k = None
    for k in range(1, len(prefix)):
        if prefix[k] >= sum_new_min:
            min_k = k
            break
    if min_k is None:
        min_k = len(unsolved_sorted)
    print(min_k)