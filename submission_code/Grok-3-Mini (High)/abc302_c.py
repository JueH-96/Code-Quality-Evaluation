import itertools

def differ_by_one(s1, s2):
    diff_count = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1

# Read input
first_line = input().split()
N = int(first_line[0])
M = int(first_line[1])  # M is read but not used in the code

# Read N strings
strings = []
for _ in range(N):
    s = input().strip()
    strings.append(s)

# Check all permutations to see if any satisfy the condition
found = False
for perm in itertools.permutations(strings):
    valid_sequence = True
    for i in range(N - 1):
        if not differ_by_one(perm[i], perm[i + 1]):
            valid_sequence = False
            break
    if valid_sequence:
        found = True
        break

# Output the result
if found:
    print("Yes")
else:
    print("No")