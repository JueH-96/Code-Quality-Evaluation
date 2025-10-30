import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + M]))

result = [-1] * M
eaten = [False] * N

for j in range(M):
    for i in range(N):
        if not eaten[i] and B[j] >= A[i]:
            result[j] = i + 1
            eaten[i] = True
            break

for res in result:
    print(res)