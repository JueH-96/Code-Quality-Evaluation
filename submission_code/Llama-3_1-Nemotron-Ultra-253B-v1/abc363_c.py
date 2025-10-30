from itertools import permutations

n, k = map(int, input().split())
s = input().strip()

unique_perms = set(permutations(s))

count = 0

for perm in unique_perms:
    valid = True
    for i in range(n - k + 1):
        is_pal = True
        for j in range(k // 2):
            if perm[i + j] != perm[i + k - 1 - j]:
                is_pal = False
                break
        if is_pal:
            valid = False
            break
    if valid:
        count += 1

print(count)