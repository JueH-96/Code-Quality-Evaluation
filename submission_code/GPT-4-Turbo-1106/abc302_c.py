from itertools import permutations

def can_transform(s1, s2):
    # Count the number of different characters
    diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
    return diff_count == 1

def can_rearrange(strings):
    for perm in permutations(strings):
        if all(can_transform(perm[i], perm[i+1]) for i in range(len(perm) - 1)):
            return True
    return False

# Read input
N, M = map(int, input().split())
strings = [input().strip() for _ in range(N)]

# Determine if the strings can be rearranged
result = can_rearrange(strings)
print("Yes" if result else "No")