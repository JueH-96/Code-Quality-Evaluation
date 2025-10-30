# YOUR CODE HERE
n, m, k = map(int, input().split())
tests = []
for i in range(m):
    line = input().split()
    c = int(line[0])
    keys = list(map(int, line[1:c + 1]))
    result = line[c + 1]
    tests.append((keys, result))

count = 0
for i in range(2 ** n):
    real_keys = []
    for j in range(n):
        if (i >> j) & 1:
            real_keys.append(j + 1)
    valid = True
    for keys, result in tests:
        num_real = 0
        for key in keys:
            if key in real_keys:
                num_real += 1
        if result == 'o' and num_real < k:
            valid = False
            break
        if result == 'x' and num_real >= k:
            valid = False
            break
    if valid:
        count += 1

print(count)