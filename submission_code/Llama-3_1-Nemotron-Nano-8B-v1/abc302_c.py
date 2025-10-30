import itertools

n, m = map(int, input().split())
strings = [input().strip() for _ in range(n)]

for perm in itertools.permutations(strings):
    valid = True
    for i in range(n - 1):
        a, b = perm[i], perm[i+1]
        diff = sum(c1 != c2 for c1, c2 in zip(a, b))
        if diff != 1:
            valid = False
            break
    if valid:
        print("Yes")
        exit()
print("No")