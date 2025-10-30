import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
C = int(data[2])
K = int(data[3])
A = list(map(int, data[4:4+N]))

# Calculate the minimum value for each k
total_sum = 0
seen = set()
current_min = float('inf')

for k in range(K):
    for a in A:
        value = (C * k + a) % M
        if value < current_min:
            current_min = value
    total_sum += current_min
    # Check for cycle detection
    state = (k % M, current_min)
    if state in seen:
        break
    seen.add(state)

print(total_sum)