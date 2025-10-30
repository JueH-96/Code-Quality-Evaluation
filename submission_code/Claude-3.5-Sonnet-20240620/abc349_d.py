# YOUR CODE HERE
def find_good_sequences(L, R):
    sequences = []
    current = L

    while current < R:
        i = 0
        while (current >> i) & 1 == 0:
            i += 1
        j = current >> i
        end = min(R, (j + 1) << i)
        sequences.append((current, end))
        current = end

    return sequences

L, R = map(int, input().split())
result = find_good_sequences(L, R)

print(len(result))
for start, end in result:
    print(start, end)