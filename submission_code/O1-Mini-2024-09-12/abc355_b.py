N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = sorted(A + B)
A_set = set(A)
found = False
for i in range(len(C)-1):
    if C[i] in A_set and C[i+1] in A_set:
        found = True
        break
print("Yes" if found else "No")