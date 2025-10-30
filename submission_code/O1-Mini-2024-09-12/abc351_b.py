n = int(input())
A = [input() for _ in range(n)]
B = [input() for _ in range(n)]
found = False
for i in range(n):
    for j in range(n):
        if A[i][j] != B[i][j]:
            print(i + 1, j + 1)
            found = True
            break
    if found:
        break