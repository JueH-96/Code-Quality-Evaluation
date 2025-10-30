n = int(input())
a = list(map(int, input().split()))

pos = [[] for _ in range(n + 1)]
for idx, num in enumerate(a):
    pos[num].append(idx + 1)  # Convert to 1-based index

result = []
for i in range(1, n + 1):
    indices = pos[i]
    indices.sort()
    f_i = indices[1]
    result.append((f_i, i))

result.sort(key=lambda x: x[0])

output = [str(x[1]) for x in result]
print(' '.join(output))