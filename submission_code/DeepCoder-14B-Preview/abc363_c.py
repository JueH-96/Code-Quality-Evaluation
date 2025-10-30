import itertools

# Read input
n, k = map(int, input().split())
s = input().strip()

# Generate all unique permutations
unique_perms = set(''.join(p) for p in itertools.permutations(s))

count = 0

for t in unique_perms:
    has_palindrome = False
    # Check each possible substring of length k
    for i in range(len(t) - k + 1):
        substr = t[i:i+k]
        if substr == substr[::-1]:
            has_palindrome = True
            break
    if not has_palindrome:
        count += 1

print(count)