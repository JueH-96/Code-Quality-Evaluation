import itertools

n, k = map(int, input().split())
r = list(map(int, input().split()))

ranges = [range(1, ri + 1) for ri in r]
sequences = itertools.product(*ranges)

valid = []
for seq in sequences:
    if sum(seq) % k == 0:
        valid.append(seq)

valid.sort()

for s in valid:
    print(' '.join(map(str, s)))