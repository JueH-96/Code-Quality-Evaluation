import itertools

n, k = map(int, input().split())
r = list(map(int, input().split()))

ranges = [range(1, ri + 1) for ri in r]

valid_sequences = []
for seq in itertools.product(*ranges):
    if sum(seq) % k == 0:
        valid_sequences.append(seq)

for seq in valid_sequences:
    print(' '.join(map(str, seq)))