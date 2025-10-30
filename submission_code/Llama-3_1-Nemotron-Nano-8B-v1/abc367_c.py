import itertools

n, k = map(int, input().split())
r = list(map(int, input().split()))

# Generate all possible sequences in lex order
ranges = [range(1, ri + 1) for ri in r]
product = itertools.product(*ranges)

valid_sequences = []
for seq in product:
    if sum(seq) % k == 0:
        valid_sequences.append(seq)

for seq in valid_sequences:
    print(' '.join(map(str, seq)))