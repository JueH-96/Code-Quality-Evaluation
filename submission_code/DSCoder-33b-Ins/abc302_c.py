from itertools import permutations

def check_strings(s1, s2):
    diff = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            diff += 1
    return diff == 1

def solve():
    N, M = map(int, input().split())
    S = [input() for _ in range(N)]

    for perm in permutations(S):
        if all(check_strings(perm[i], perm[i+1]) for i in range(N-1)):
            print("Yes")
            return
    print("No")

solve()