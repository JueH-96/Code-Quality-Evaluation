# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    P = list(map(int, data[index + 1:index + N + 1]))
    index += N + 1
    
    operations = 0
    for i in range(N):
        if P[i] != i + 1:
            operations += 1
            if i > 0:
                P[:i] = sorted(P[:i])
            if i < N - 1:
                P[i + 1:] = sorted(P[i + 1:])
    
    results.append(operations)

for result in results:
    print(result)