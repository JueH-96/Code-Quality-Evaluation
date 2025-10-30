import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
P = []
T = []
for i in range(N-1):
    P.append(int(data[3+2*i]))
    T.append(int(data[4+2*i]))
Q = int(data[3+2*(N-1)])
q = [int(data[4+2*(N-1)+i]) for i in range(Q)]

def find_next_bus_time(current_time, bus_stop):
    if bus_stop == 0:
        return X
    next_departure = (current_time // P[bus_stop-1] + 1) * P[bus_stop-1]
    return next_departure + T[bus_stop-1]

def find_earliest_arrival_time(q):
    result = []
    for qi in q:
        current_time = qi
        bus_stop = 0
        while bus_stop < N-1:
            next_bus_time = find_next_bus_time(current_time, bus_stop)
            if next_bus_time <= current_time + X:
                bus_stop += 1
                current_time = next_bus_time
            else:
                break
        if bus_stop == N-1:
            result.append(current_time + X + Y)
        else:
            result.append(current_time + X)
    return result

earliest_arrival_times = find_earliest_arrival_time(q)
for time in earliest_arrival_times:
    print(time)