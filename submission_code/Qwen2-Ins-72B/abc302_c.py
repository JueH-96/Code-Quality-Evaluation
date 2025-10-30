from itertools import permutations

N, M = map(int, input().split())
S = [input() for _ in range(N)]

def check_sequence(seq):
    for i in range(N-1):
        diff = 0
        for j in range(M):
            if seq[i][j] != seq[i+1][j]:
                diff += 1
        if diff != 1:
            return False
    return True

for perm in permutations(S):
    if check_sequence(perm):
        print("Yes")
        exit()

print("No")