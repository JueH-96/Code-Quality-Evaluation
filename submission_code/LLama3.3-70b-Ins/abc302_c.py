from itertools import permutations

def is_valid_permutation(perm):
    for i in range(len(perm) - 1):
        diff_count = sum(c1 != c2 for c1, c2 in zip(perm[i], perm[i + 1]))
        if diff_count != 1:
            return False
    return True

def solve():
    n, m = map(int, input().split())
    strings = [input() for _ in range(n)]

    for perm in permutations(strings):
        if is_valid_permutation(perm):
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()