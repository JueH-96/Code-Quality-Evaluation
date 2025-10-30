N, M = map(int, input().split())
S = input().strip()
T = input().strip()

possible = True
for i in range(N):
    found = False
    for j in range(M):
        s = i - j
        if s >= 0 and s + M <= N and T[j] == S[i]:
            found = True
            break
    if not found:
        possible = False
        break

print("Yes" if possible else "No")