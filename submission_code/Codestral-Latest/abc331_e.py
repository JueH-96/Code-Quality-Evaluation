# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
L = int(data[2])

a = list(map(int, data[3:3+N]))
b = list(map(int, data[3+N:3+N+M]))

not_offered = set()
for i in range(L):
    c = int(data[3+N+M+2*i]) - 1
    d = int(data[3+N+M+2*i+1]) - 1
    not_offered.add((c, d))

max_price = 0
for i in range(N):
    for j in range(M):
        if (i, j) not in not_offered:
            max_price = max(max_price, a[i] + b[j])

print(max_price)