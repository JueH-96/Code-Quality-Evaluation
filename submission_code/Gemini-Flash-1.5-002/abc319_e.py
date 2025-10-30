# YOUR CODE HERE
N, X, Y = map(int, input().split())
P = []
T = []
for i in range(N - 1):
    p, t = map(int, input().split())
    P.append(p)
    T.append(t)
Q = int(input())
for i in range(Q):
    q = int(input())
    time = q + X
    for j in range(N - 1):
        next_time = (time + P[j] - 1) // P[j] * P[j]
        time = next_time + T[j]
    time += Y
    print(time)