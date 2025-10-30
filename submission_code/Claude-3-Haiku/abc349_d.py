from math import log2, floor

def find_good_sequences(L, R):
    sequences = []
    i = 0
    while L < R:
        j = floor(log2(L))
        while 2 ** (j + 1) * (j + 1) <= R:
            j += 1
        r = min(2 ** j * (j + 1), R)
        sequences.append((L, r))
        L = r
    return sequences

L, R = map(int, input().split())
sequences = find_good_sequences(L, R)
print(len(sequences))
for l, r in sequences:
    print(l, r)