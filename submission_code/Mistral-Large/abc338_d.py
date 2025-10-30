import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))

    # Calculate the minimum length of the tour when each bridge is closed
    min_length = float('inf')

    # Check closing each bridge from 1 to N-1
    for i in range(1, N):
        length = 0
        current = X[0]

        for j in range(1, M):
            target = X[j]
            if current < target <= i or target < current <= i or (current > i and target <= i):
                length += min(abs(target - current), N - abs(target - current))
            else:
                length += abs(target - current)
            current = target

        min_length = min(min_length, length)

    # Check closing the N-th bridge
    length = 0
    current = X[0]

    for j in range(1, M):
        target = X[j]
        if current == 1 and target == N:
            length += 1
        elif current == N and target == 1:
            length += 1
        else:
            length += abs(target - current)
        current = target

    min_length = min(min_length, length)

    print(min_length)

solve()