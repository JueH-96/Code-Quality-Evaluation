from itertools import permutations

def differ_by_one(s1, s2):
    if len(s1) != len(s2):
        return False
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
    return diff_count == 1

N, M = map(int, input().split())
strings = []
for _ in range(N):
    strings.append(input().strip())

# Try all permutations
found = False
for perm in permutations(strings):
    valid = True
    for i in range(N - 1):
        if not differ_by_one(perm[i], perm[i+1]):
            valid = False
            break
    if valid:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")