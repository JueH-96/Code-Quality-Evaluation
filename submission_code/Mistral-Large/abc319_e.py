import sys
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

def calculate_time(start_time):
    current_time = start_time + X
    for i in range(N-1):
        if current_time % P[i] == 0:
            current_time += T[i]
        else:
            next_bus = ((current_time + P[i] - 1) // P[i]) * P[i]
            current_time = next_bus + T[i]
    current_time += Y
    return current_time

results = [calculate_time(q) for q in queries]

sys.stdout.write("
".join(map(str, results)) + "
")