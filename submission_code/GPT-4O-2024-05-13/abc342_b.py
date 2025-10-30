# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
P = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = data[N+2:]

# Create a dictionary to map person number to their position
position_map = {P[i]: i+1 for i in range(N)}

results = []
for i in range(Q):
    A = int(queries[2*i])
    B = int(queries[2*i+1])
    if position_map[A] < position_map[B]:
        results.append(A)
    else:
        results.append(B)

for result in results:
    print(result)