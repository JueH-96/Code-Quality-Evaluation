import sys
from itertools import permutations

def min_operations_to_match(N, S, T):
    # Generate all possible permutations of the stones
    perms = permutations(S)

    # Check each permutation to see if it matches T
    for perm in perms:
        perm_str = ''.join(perm)
        if perm_str == T:
            # Calculate the minimum number of operations required
            operations = 0
            for i in range(N):
                if perm[i] != S[i]:
                    # Find the position of the stone in S that matches T[i]
                    pos = S.index(T[i])
                    # Move the stone to the correct position
                    S = S[:pos] + S[pos+1:] + S[pos]
                    operations += 1
            return operations

    return -1

# Read input
N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Calculate and print the result
result = min_operations_to_match(N, S, T)
print(result)