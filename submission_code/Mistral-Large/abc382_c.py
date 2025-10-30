import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:2+N+M]))

for b in B:
    eaten = False
    for i in range(N):
        if b >= A[i]:
            print(i + 1)
            eaten = True
            break
    if not eaten:
        print(-1)