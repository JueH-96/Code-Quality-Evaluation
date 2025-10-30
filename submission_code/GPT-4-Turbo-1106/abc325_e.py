import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def min_travel_time(N, A, B, C, D):
    # Initialize the minimum time to travel from city 1 to city N
    min_time = [float('inf')] * N
    min_time[0] = 0  # Starting city

    # Calculate the minimum time to travel using company car
    for i in range(N):
        for j in range(i + 1, N):
            min_time[j] = min(min_time[j], min_time[i] + D[i][j] * A)

    # Calculate the minimum time to travel using train from any city
    train_time = min_time[:]
    for i in range(N):
        train_time[i] += D[i][N - 1] * B + C

    # The answer is the minimum of the last city time using car or train
    return min(min_time[N - 1], min(train_time))

# Read input
N, A, B, C = read_ints()
D = [read_ints() for _ in range(N)]

# Solve the problem and print the answer
print(min_travel_time(N, A, B, C, D))