import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = list(map(int, data[1:N+1]))
Q = int(data[N+1])
tasks = [(int(data[N+2+2*i])-1, int(data[N+3+2*i])) for i in range(Q)]

# Calculate the minimum number of movements required
total_moves = 0
for person, target in tasks:
    if X[person] < target:
        moves = target - X[person]
        X[person] += moves
    else:
        moves = X[person] - target
        X[person] -= moves
    total_moves += moves

print(total_moves)