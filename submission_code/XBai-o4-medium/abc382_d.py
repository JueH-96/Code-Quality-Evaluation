n, m = map(int, input().split())
result = []

def dfs(current, pos):
    if pos > n:
        result.append(current.copy())
        return
    if pos == 1:
        start = 1
    else:
        start = current[-1] + 10
    max_val = m - 10 * (n - pos)
    for val in range(start, max_val + 1):
        current.append(val)
        dfs(current, pos + 1)
        current.pop()

dfs([], 1)

print(len(result))
for seq in result:
    print(' '.join(map(str, seq)))