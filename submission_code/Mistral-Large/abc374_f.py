import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    T = list(map(int, data[3:]))

    dissatisfaction = 0
    current_day = 0
    i = 0

    while i < N:
        # Determine the next possible shipping day
        next_ship_day = max(current_day + X, T[i])

        # Ship as many orders as possible on the next_ship_day
        j = i
        while j < N and j < i + K and T[j] <= next_ship_day:
            dissatisfaction += (next_ship_day - T[j])
            j += 1

        # Update the current day and index
        current_day = next_ship_day
        i = j

    print(dissatisfaction)

solve()