n, m = map(int, input().split())
wheels = []
for _ in range(n):
    parts = list(map(int, input().split()))
    c = parts[0]
    p = parts[1]
    s = parts[2:2+p]
    wheels.append((c, s))

E = [0.0] * (m + 1)
for s in range(m-1, -1, -1):
    min_cost = float('inf')
    for (c, s_list) in wheels:
        total = 0.0
        for sj in s_list:
            new_s = s + sj
            if new_s >= m:
                total += 0.0
            else:
                total += E[new_s]
        cost = c + total / len(s_list)
        if cost < min_cost:
            min_cost = cost
    E[s] = min_cost

print("{0:.10f}".format(E[0]))