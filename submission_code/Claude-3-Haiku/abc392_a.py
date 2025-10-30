import itertools

# Read input from stdin
A1, A2, A3 = map(int, input().split())

# Generate all permutations of the input sequence
perms = list(itertools.permutations([A1, A2, A3]))

# Check if any permutation satisfies the condition
for perm in perms:
    if perm[0] * perm[1] == perm[2]:
        print("Yes")
        break
else:
    print("No")