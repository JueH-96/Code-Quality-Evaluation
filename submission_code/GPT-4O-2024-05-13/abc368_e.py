# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X1 = int(data[2])

trains = []
index = 3
for i in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    S = int(data[index + 2])
    T = int(data[index + 3])
    trains.append((A, B, S, T))
    index += 4

# Initialize X array with X1 for the first train and 0 for others
X = [0] * M
X[0] = X1

# Create a graph to represent the constraints
graph = [[] for _ in range(M)]
for i in range(M):
    for j in range(M):
        if i != j and trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
            graph[i].append(j)

# Use a priority queue to process the constraints
pq = []
for i in range(M):
    heapq.heappush(pq, (trains[i][3] + X[i], i))

while pq:
    current_time, i = heapq.heappop(pq)
    for j in graph[i]:
        required_time = current_time
        if required_time > trains[j][2] + X[j]:
            X[j] = required_time - trains[j][2]
            heapq.heappush(pq, (trains[j][3] + X[j], j))

# Output the result
print(" ".join(map(str, X[1:])))