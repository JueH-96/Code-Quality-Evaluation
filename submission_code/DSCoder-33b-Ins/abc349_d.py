# YOUR CODE HERE

def find_good_sequences(L, R):
    i = 0
    while 2**i * L < 2**i * (L+1) <= R:
        i += 1
    i -= 1
    sequences = []
    while L < R:
        j = L
        while 2**i * j < 2**i * (j+1) <= R:
            j += 1
        j -= 1
        sequences.append((2**i * L, 2**i * (j+1)))
        L = 2**i * (j+1)
        while 2**(i-1) * L < 2**(i-1) * (L+1) <= R:
            i -= 1
    return sequences

L, R = map(int, input().split())
sequences = find_good_sequences(L, R)
print(len(sequences))
for sequence in sequences:
    print(*sequence)