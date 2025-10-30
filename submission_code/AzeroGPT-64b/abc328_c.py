from itertools import pairwise

N, Q = map(int, input().split())
S = input()
S_pairs = [int(a == b) for a, b in pairwise(S)]
acc = list(itertools.accumulate(S_pairs))
for _ in range(Q):
    l, r = map(int, input().split())
    print((acc[r - 1] - acc[l - 1] + S_pairs[l - 1] if l >= 1 else acc[r - 1]))