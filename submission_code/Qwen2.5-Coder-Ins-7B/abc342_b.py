# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = [(int(data[N+2+2*i])-1, int(data[N+3+2*i])-1) for i in range(Q)]

for A, B in queries:
    if P[A] < P[B]:
        print(P[A])
    else:
        print(P[B])