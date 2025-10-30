import itertools

n, m = map(int, input().split())
strings = [input().strip() for _ in range(n)]

for perm in itertools.permutations(strings):
    valid = True
    for i in range(len(perm) - 1):
        s1 = perm[i]
        s2 = perm[i + 1]
        diff = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        if diff != 1:
            valid = False
            break
    if valid:
        print("Yes")
        exit()
print("No")