import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])

A = list(map(int, data[3:N+3]))
B = list(map(int, data[N+3:]))

total_sweetness = 0
total_saltiness = 0
dishes_eaten = 0

for i in range(N):
    total_sweetness += A[i]
    total_saltiness += B[i]
    dishes_eaten += 1
    if total_sweetness > X or total_saltiness > Y:
        break

print(dishes_eaten)