# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

W = int(data[0])
H = int(data[1])
N = int(data[2])
p = list(map(int, data[3*N:4*N]))
q = list(map(int, data[4*N:5*N]))
A = int(data[5*N])
a = list(map(int, data[5*N+1:5*N+1+A]))
B = int(data[5*N+1+A])
b = list(map(int, data[5*N+2+A:5*N+2+A+B]))

p.append(0)
p.append(W)
q.append(0)
q.append(H)
p.sort()
q.sort()

min_strawberries = float('inf')
max_strawberries = 0
for i in range(1, len(p)):
    for j in range(1, len(q)):
        strawberries = 0
        for x, y in zip(p[:-1], p[1:]):
            if x > a[0] and x < a[-1]:
                continue
            for y1, y2 in zip(q[:-1], q[1:]):
                if y1 > b[0] and y1 < b[-1]:
                    continue
                if x > a[0] and x < a[-1]:
                    continue
                if y1 > b[0] and y1 < b[-1]:
                    continue
                for k in range(N):
                    if p[i-1] < p[k] < p[i] and q[j-1] < q[k] < q[j]:
                        strawberries += 1
        min_strawberries = min(min_strawberries, strawberries)
        max_strawberries = max(max_strawberries, strawberries)

print(min_strawberries, max_strawberries)