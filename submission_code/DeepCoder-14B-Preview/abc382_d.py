n, m = map(int, input().split())

sequences = []

def backtrack(current, i):
    if i == n + 1:
        sequences.append(current.copy())
        return
    if i == 1:
        start = 1
    else:
        start = current[-1] + 10
    if i < n:
        steps_remaining = n - i
        end = m - 10 * steps_remaining
    else:
        end = m
    for val in range(start, end + 1):
        current.append(val)
        backtrack(current, i + 1)
        current.pop()

backtrack([], 1)

print(len(sequences))
for seq in sequences:
    print(' '.join(map(str, seq)))