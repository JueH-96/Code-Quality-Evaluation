n = int(input())
A = list(map(int, input().split()))

# Precompute the list of pairs
pairs = []
i = 0
while i < len(A) - 1:
    if A[i] == A[i + 1]:
        pairs.append((A[i], i, i + 1))
        i += 2
    else:
        i += 1

max_pairs = 0

for i in range(len(pairs)):
    used = set()
    current = 0
    for j in range(i, len(pairs)):
        val = pairs[j][0]
        if val not in used:
            used.add(val)
            current += 1
        else:
            break
    if current > max_pairs:
        max_pairs = current

max_length = 2 * max_pairs
print(max_length if max_length >= 2 else 0)