import itertools

n, k = map(int, input().split())
s = input().strip()

# Generate all unique permutations
unique_perms = set()
for p in itertools.permutations(s):
    unique_perms.add(''.join(p))

count = 0
for perm in unique_perms:
    valid = True
    for i in range(n - k + 1):
        substring = perm[i:i+k]
        if substring == substring[::-1]:
            valid = False
            break
    if valid:
        count += 1

print(count)