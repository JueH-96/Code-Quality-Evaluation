from itertools import permutations

def can_rearrange(N, M, strings):
    # Generate all possible permutations of the strings
    for perm in permutations(strings):
        valid = True
        for i in range(N-1):
            # Compare current and next string in the permutation
            diff = 0
            for j in range(M):
                if perm[i][j] != perm[i+1][j]:
                    diff += 1
                    if diff > 1:
                        break
            if diff != 1:
                valid = False
                break
        if valid:
            return True
    return False

# Read input
N, M = map(int, input().split())
strings = [input().strip() for _ in range(N)]

# Check if rearrangement is possible
if can_rearrange(N, M, strings):
    print("Yes")
else:
    print("No")