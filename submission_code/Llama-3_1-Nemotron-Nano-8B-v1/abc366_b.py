n = int(input())
strings = [input().strip() for _ in range(n)]
m = max(len(s) for s in strings)

result = []
for j in range(1, m + 1):
    max_p = 0
    for i in range(1, n + 1):
        if j <= len(strings[i - 1]):
            p_candidate = n - i + 1
            if p_candidate > max_p:
                max_p = p_candidate
    t = []
    for p in range(1, max_p + 1):
        current_i = n - p + 1
        if 1 <= current_i <= n and j <= len(strings[current_i - 1]):
            t.append(strings[current_i - 1][j - 1])
        else:
            t.append('*')
    result.append(''.join(t))

for line in result:
    print(line)