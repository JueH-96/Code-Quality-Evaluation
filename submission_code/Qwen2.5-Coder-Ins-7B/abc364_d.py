import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

a = list(map(int, data[2:2+N]))
b = list(map(int, data[2+N:2+N+Q]))
k = list(map(int, data[2+N+Q:]))

for i in range(Q):
    distances = sorted([abs(a[j] - b[i]) for j in range(N)])
    print(distances[k[i]-1])