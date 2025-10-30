from itertools import permutations

n, k = map(int, input().split())
s = input().strip()

# Generate all unique permutations of the string s
unique_perms = set()
for p in permutations(s):
    unique_perms.add(''.join(p))

count = 0

# Check each permutation for palindromic substrings of length k
for perm in unique_perms:
    valid = True
    # Check all possible starting positions for substrings of length k
    for i in range(len(perm) - k + 1):
        substr = perm[i:i+k]
        if substr == substr[::-1]:
            valid = False
            break
    if valid:
        count += 1

print(count)