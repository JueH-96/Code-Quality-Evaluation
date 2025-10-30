import itertools

n, m = map(int, input().split())
strings = [input().strip() for _ in range(n)]

for perm in itertools.permutations(strings):
    valid = True
    for i in range(n - 1):
        a = perm[i]
        b = perm[i + 1]
        diff = 0
        for j in range(m):
            if a[j] != b[j]:
                diff += 1
        if diff != 1:
            valid = False
            break
    if valid:
        print("Yes")
        exit()
print("No")