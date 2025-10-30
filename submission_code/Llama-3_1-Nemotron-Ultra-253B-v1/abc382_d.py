n, m = map(int, input().split())
result = []

def backtrack(seq):
    if len(seq) == n:
        result.append(seq)
        return
    i = len(seq) + 1
    if i == 1:
        min_val = 1
    else:
        min_val = seq[-1] + 10
    max_val = m - 10 * (n - i)
    if min_val > max_val:
        return
    for val in range(min_val, max_val + 1):
        backtrack(seq + [val])

backtrack([])
print(len(result))
for seq in result:
    print(' '.join(map(str, seq)))