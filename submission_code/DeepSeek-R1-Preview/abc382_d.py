n, m = map(int, input().split())
sequences = []

def backtrack(current, step):
    if step == n + 1:
        sequences.append(current.copy())
        return
    if step == 1:
        lower = 1
        upper = m - 10 * (n - step)
    else:
        lower = current[-1] + 10
        upper = m - 10 * (n - step)
    if lower > upper:
        return
    if step == 1 and upper < lower:
        return
    for ai in range(lower, upper + 1):
        current.append(ai)
        backtrack(current, step + 1)
        current.pop()

backtrack([], 1)

print(len(sequences))
for seq in sequences:
    print(' '.join(map(str, seq)))