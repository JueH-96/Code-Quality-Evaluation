N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Merge A and B while maintaining order
C = []
i, j = 0, 0
while i < N and j < M:
    if A[i] < B[j]:
        C.append(A[i])
        i += 1
    else:
        C.append(B[j])
        j += 1
C += A[i:]
C += B[j:]

# Check if two consecutive elements in C are from A
prev_from_A = False
for c in C:
    if c in A:
        if prev_from_A:
            print("Yes")
            exit()
        prev_from_A = True
    else:
        prev_from_A = False
print("No")