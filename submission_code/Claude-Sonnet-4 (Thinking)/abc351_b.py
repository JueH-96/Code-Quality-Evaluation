N = int(input())

# Read grid A
A = []
for i in range(N):
    A.append(input())

# Read grid B
B = []
for i in range(N):
    B.append(input())

# Find the difference
found = False
for i in range(N):
    for j in range(N):
        if A[i][j] != B[i][j]:
            print(i+1, j+1)
            found = True
            break
    if found:
        break