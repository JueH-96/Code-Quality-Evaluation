from itertools import product

N, M, K = map(int, input().split())

doors = []
for _ in range(M):
    C, *A, R = input().split()
    C = int(C)
    A = [int(a) - 1 for a in A]
    doors.append((C, A, R))

ans = 0
for keys in product([0, 1], repeat=N):
    for C, A, R in doors:
        if (R == "o" and sum(keys[a] for a in A) < K) or (R == "x" and sum(keys[a] for a in A) >= K):
            break
    else:
        ans += 1

print(ans)