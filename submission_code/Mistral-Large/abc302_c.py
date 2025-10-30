import sys
from itertools import permutations

def can_transform(s1, s2):
    differences = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            differences += 1
        if differences > 1:
            return False
    return differences == 1

def solve(N, M, strings):
    for perm in permutations(strings):
        valid = True
        for i in range(N - 1):
            if not can_transform(perm[i], perm[i + 1]):
                valid = False
                break
        if valid:
            return "Yes"
    return "No"

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    strings = data[2:]

    result = solve(N, M, strings)
    print(result)

if __name__ == "__main__":
    main()