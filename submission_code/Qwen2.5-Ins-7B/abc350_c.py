# N = int(input())
# A = list(map(int, input().split()))

def transform_sequence(N, A):
    operations = []
    for i in range(N):
        if A[i] != i + 1:
            for j in range(i + 1, N):
                if A[j] == i + 1:
                    operations.append((i + 1, j + 1))
                    A[i], A[j] = A[j], A[i]
                    break
    print(len(operations))
    for op in operations:
        print(*op)

N = int(input())
A = list(map(int, input().split()))
transform_sequence(N, A)