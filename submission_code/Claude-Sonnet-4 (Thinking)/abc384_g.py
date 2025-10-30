N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())

for _ in range(K):
    X, Y = map(int, input().split())
    total = 0
    for i in range(X):
        for j in range(Y):
            total += abs(A[i] - B[j])
    print(total)