import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1:]

row_count = [0] * N
col_count = [0] * N

for i in range(N):
    for j in range(N):
        if S[i][j] == 'o':
            row_count[i] += 1
            col_count[j] += 1

answer = 0

for i in range(N):
    for j in range(N):
        if S[i][j] == 'o':
            answer += (row_count[i] - 1) * (col_count[j] - 1)

print(answer)