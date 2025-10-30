# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

colors = list(map(int, data[2:2+N]))
queries = list(zip(map(int, data[2+N:2+N+2*Q:2]), map(int, data[2+N+1:2+N+2*Q:2])))

for a, b in queries:
    colors[b-1:] = colors[a-1:b] + colors[b-1:]
    print(len(set(colors[b-1:])))