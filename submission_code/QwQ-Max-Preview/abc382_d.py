n, m = map(int, input().split())

sequences = []

def backtrack(current):
    if len(current) == n:
        sequences.append(current.copy())
        return
    if len(current) == 0:
        min_val = 1
        remaining = n - 1
        max_val = m - 10 * remaining
    else:
        min_val = current[-1] + 10
        remaining = n - (len(current) + 1)
        max_val = m - 10 * remaining
    for a in range(min_val, max_val + 1):
        current.append(a)
        backtrack(current)
        current.pop()

backtrack([])
print(len(sequences))
for seq in sequences:
    print(' '.join(map(str, seq)))