import bisect
from collections import defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    X1 = int(input[idx]); idx += 1

    trains = [(0, 0, 0, 0)]  # trains[0] dummy
    for j in range(1, M + 1):
        A = int(input[idx]); idx += 1
        B = int(input[idx]); idx += 1
        S = int(input[idx]); idx += 1
        T = int(input[idx]); idx += 1
        trains.append((A, B, S, T))

    # Generate events
    events = []
    for j in range(1, M + 1):
        A, B, S, T = trains[j]
        events.append((S, 'D', A, j))  # Departure event
        events.append((T, 'A', B, j))  # Arrival event

    # Sort events by time, arrivals before departures at the same time
    def event_key(event):
        time, typ, city, j = event
        return (time, 0 if typ == 'A' else 1)
    events.sort(key=event_key)

    # Initialize U array
    U = [0] * (M + 1)
    A1, B1, S1, T1 = trains[1]
    U[1] = S1 + X1

    # Data structure for each city: times and prefix_max
    city_data = defaultdict(lambda: {'times': [], 'prefix_max': [-float('inf')]})

    for event in events:
        time, typ, city, j = event
        if typ == 'A':
            # Arrival of train j, update its city's data
            _, _, S_j, T_j = trains[j]
            D_j = T_j - S_j
            val = U[j] + D_j
            times = city_data[city]['times']
            prefix_max = city_data[city]['prefix_max']
            times.append(time)
            new_max = max(prefix_max[-1], val)
            prefix_max.append(new_max)
        else:
            if j == 1:
                continue  # Skip processing departure of train 1
            # Departure of train j, compute U[j]
            C = city
            S_j = time
            times = city_data[C]['times']
            prefix_max = city_data[C]['prefix_max']
            pos = bisect.bisect_right(times, S_j)
            if pos == 0:
                max_val = -float('inf')
            else:
                max_val = prefix_max[pos]
            U[j] = max(S_j, max_val)

    # Collect results X_2 to X_M
    result = []
    for j in range(2, M + 1):
        S_j = trains[j][2]
        result.append(str(U[j] - S_j))
    print(' '.join(result))

if __name__ == '__main__':
    main()