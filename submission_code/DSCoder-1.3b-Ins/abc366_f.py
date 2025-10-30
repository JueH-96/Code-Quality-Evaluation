N, K = map(int, input().split())
functions = [tuple(map(int, input().split())) for _ in range(N)]

max_value = 0
for p in combinations(range(N), K):
    f_p = [functions[i] for i in p]
    value = f_p[0][0] * f_p[1][1] + f_p[0][1] * f_p[1][0]
    for i in range(2, len(f_p)):
        value *= f_p[i][0] * f_p[i][1]
    max_value = max(max_value, value)

print(max_value)