import sys
from itertools import permutations

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    N, M = map(int, lines[0].split())
    conditions = []
    for line in lines[1:]:
        L, R, X = map(int, line.split())
        conditions.append((L-1, R-1, X-1))
    return N, conditions

def check_permutation(permutation, conditions):
    """Check if a permutation satisfies all conditions."""
    for L, R, X in conditions:
        max_val = max(permutation[L:R+1])
        if max_val == permutation[X]:
            return False
    return True

def count_permutations(N, conditions):
    """Count the number of permutations that satisfy all conditions."""
    count = 0
    for permutation in permutations(range(N)):
        if check_permutation(permutation, conditions):
            count += 1
    return count % 998244353

def main():
    N, conditions = read_input()
    result = count_permutations(N, conditions)
    print(result)

if __name__ == "__main__":
    main()