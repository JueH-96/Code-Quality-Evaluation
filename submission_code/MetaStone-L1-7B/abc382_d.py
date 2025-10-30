def backtrack(pos, current, last_val):
    if pos == N:
        if last_val <= M:
            yield current.copy()
        return
    else:
        min_next = last_val + 10
        remaining_steps = N - pos - 1
        max_next = M - 10 * remaining_steps
        if min_next > max_next:
            return
        for next_val in range(min_next, max_next + 1):
            current.append(next_val)
            yield from backtrack(pos + 1, current, next_val)
            current.pop()

n, m = map(int, input().split())

max_a1 = m - 10 * (n - 1)
if max_a1 < 1:
    print(0)
else:
    sequences = []
    for a in range(1, max_a1 + 1):
        current = [a]
        for seq in backtrack(1, current, a):
            sequences.append(seq)
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))