n, x = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
p = list(map(int, input().split()))
q = list(map(int, input().split()))

x -= 1

visited = [False] * n
count = 0
for i in range(n):
    if a[i] == 0 and b[i] == 0 and i != x:
        continue
    if visited[i]:
        continue
    cycle = []
    curr = i
    while not visited[curr]:
        visited[curr] = True
        cycle.append(curr)
        curr = p[curr] - 1 if a[curr] == 1 else q[curr] - 1
    
    cycle_len = len(cycle)
    
    
    has_x = False
    for j in cycle:
        if j == x:
            has_x = True
            break
    
    if not has_x:
        
        all_zero = True
        for j in cycle:
            if a[j] != 0 or b[j] != 0:
                all_zero = False
                break
        if not all_zero:
            print(-1)
            exit()
    else:
        count += cycle_len

print(count)