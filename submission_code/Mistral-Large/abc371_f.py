import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = list(map(int, data[1:N+1]))
Q = int(data[N+1])
tasks = []
for i in range(Q):
    T = int(data[N+2+2*i])
    G = int(data[N+2+2*i+1])
    tasks.append((T, G))

positions = X[:]
total_moves = 0

for T, G in tasks:
    current_pos = positions[T-1]
    if current_pos < G:
        for i in range(T-1, N):
            if positions[i] < G:
                total_moves += G - positions[i]
                positions[i] = G
    elif current_pos > G:
        for i in range(T-1, -1, -1):
            if positions[i] > G:
                total_moves += positions[i] - G
                positions[i] = G

print(total_moves)