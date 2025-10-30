from itertools import permutations

# Read input
a = list(map(int, input().split()))

# Check all permutations
for p in permutations(a):
    if p[0] * p[1] == p[2]:
        print("Yes")
        exit()

# If none found
print("No")