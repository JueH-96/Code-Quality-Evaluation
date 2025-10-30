import sys

N = int(input())
A = list(map(int, input().split()))
Rs = {}
L = []
for i, a in enumerate(A):
    if a not in Rs:
        Rs[a] = i
        L.append((i, a))
    elif a in Rs and a not in Ds:
        Ds[a] = i
    else:
        Ts[a] = i - (Ds[a] - Rs[a]) - Rs[a]
        L.remove((Rs[a], a))
L.sort(key=lambda R: Ts[R[1]])
print(*sorted(set(map(lambda R: R[1], L))))