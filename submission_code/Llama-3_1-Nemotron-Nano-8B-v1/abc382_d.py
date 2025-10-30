n, m = map(int, input().split())

max_x1 = m - 10 * (n - 1)
result = []

for x1 in range(1, max_x1 + 1):
    K = m - x1 - 10 * (n - 1)
    if K < 0:
        continue
    current_d = []
    
    def backtrack_d(pos, remaining):
        if pos == n - 1:
            seq = [x1]
            for i in range(n - 1):
                seq.append(seq[-1] + 10 + current_d[i])
            result.append(seq)
            return
        for d in range(0, remaining + 1):
            current_d.append(d)
            backtrack_d(pos + 1, remaining - d)
            current_d.pop()
    
    backtrack_d(0, K)

print(len(result))
for seq in result:
    print(' '.join(map(str, seq)))