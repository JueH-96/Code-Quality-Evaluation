import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + N]))
index += N
K = int(data[index])
index += 1
XY = [list(map(int, data[index:index + 2])) for index in range(index, index + 2 * K, 2)]

for X_k, Y_k in XY:
    result = sum(abs(A[i] - B[j]) for i in range(X_k) for j in range(Y_k))
    print(result)