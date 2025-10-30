from math import ceil

# Read input
N, X, Y = map(int, input().split())
P = []
T = []
for _ in range(N-1):
    p, t = map(int, input().split())
    P.append(p)
    T.append(t)
Q = int(input())
q = [int(input()) for _ in range(Q)]

# Function to calculate the earliest arrival time
def earliest_arrival(start_time):
    time = start_time + X
    for i in range(N-1):
        wait_time = (ceil(time / P[i]) * P[i]) - time
        time += wait_time + T[i]
    time += Y
    return time

# Print the answers
for qi in q:
    print(int(earliest_arrival(qi)))