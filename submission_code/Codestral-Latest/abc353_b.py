import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

count = 0
seats = K

for i in range(N):
    if seats < A[i]:
        count += 1
        seats = K
    seats -= A[i]

print(count)