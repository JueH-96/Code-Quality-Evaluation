import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
X = list(map(int, data[index:index + N]))
index += N
Q = int(data[index])
index += 1
tasks = [tuple(map(int, data[index:index + 2])) for index in range(index, index + 2 * Q, 2)]

# Calculate the minimum movements required
def calculate_movements(N, X, Q, tasks):
    # Dictionary to store the current position of each person
    positions = {i: X[i - 1] for i in range(1, N + 1)}

    # Dictionary to store the target positions for each person
    targets = defaultdict(list)

    for T, G in tasks:
        targets[T].append(G)

    total_movements = 0

    for person in range(1, N + 1):
        if person in targets:
            targets[person].sort()
            current_pos = positions[person]
            for target in targets[person]:
                total_movements += abs(target - current_pos)
                current_pos = target
            positions[person] = current_pos

    return total_movements

result = calculate_movements(N, X, Q, tasks)
print(result)