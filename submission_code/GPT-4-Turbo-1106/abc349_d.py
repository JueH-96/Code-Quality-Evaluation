def find_good_sequences(L, R):
    sequences = []
    while L < R:
        x = L
        while x & 1 == 0:
            x >>= 1
        i = 0
        while (L + (1 << i)) <= R and ((L >> i) & 1) == 0:
            i += 1
        sequences.append((L, L + (1 << (i - 1))))
        L += (1 << (i - 1))
    return sequences

L, R = map(int, input().split())
good_sequences = find_good_sequences(L, R)
print(len(good_sequences))
for l, r in good_sequences:
    print(l, r)