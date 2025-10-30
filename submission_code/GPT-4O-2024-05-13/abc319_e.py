import sys
import math

input = sys.stdin.read
data = input().split()

index = 0

N = int(data[index])
index += 1
X = int(data[index])
index += 1
Y = int(data[index])
index += 1

P = []
T = []

for i in range(N-1):
    P.append(int(data[index]))
    index += 1
    T.append(int(data[index]))
    index += 1

Q = int(data[index])
index += 1

queries = []
for i in range(Q):
    queries.append(int(data[index]))
    index += 1

results = []

for q in queries:
    current_time = q + X  # Time when Takahashi reaches bus stop 1
    for i in range(N-1):
        # Wait for the next bus
        if current_time % P[i] == 0:
            next_bus_time = current_time
        else:
            next_bus_time = (current_time // P[i] + 1) * P[i]
        current_time = next_bus_time + T[i]
    current_time += Y  # Time to walk from bus stop N to Aoki's house
    results.append(current_time)

for result in results:
    print(result)