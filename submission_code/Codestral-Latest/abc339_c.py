import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize the minimum number of passengers
min_passengers = 0
current_passengers = 0

# Iterate through the stops
for i in range(N):
    current_passengers += A[i]
    if current_passengers < min_passengers:
        min_passengers = current_passengers

# The minimum possible current number of passengers
print(-min_passengers)