import sys

def calculate_arrival_time(N, X, Y, P, T, Q, queries):
    """
    Calculate the arrival time for each query.

    Args:
    N (int): The number of bus stops.
    X (int): The time it takes to walk from the house to the first bus stop.
    Y (int): The time it takes to walk from the last bus stop to Aoki's house.
    P (list): A list of integers representing the time interval for each bus.
    T (list): A list of integers representing the time it takes for each bus to travel.
    Q (int): The number of queries.
    queries (list): A list of integers representing the time Takahashi leaves his house for each query.

    Returns:
    list: A list of integers representing the arrival time for each query.
    """
    arrival_times = []
    for query in queries:
        time = query + X
        for i in range(N - 1):
            # Calculate the next bus departure time
            next_departure = (time // P[i] + 1) * P[i]
            # Update the time to the next bus departure time plus the travel time
            time = next_departure + T[i]
        # Add the time it takes to walk from the last bus stop to Aoki's house
        time += Y
        arrival_times.append(time)
    return arrival_times

def main():
    # Read input from stdin
    N, X, Y = map(int, sys.stdin.readline().split())
    P = []
    T = []
    for _ in range(N - 1):
        p, t = map(int, sys.stdin.readline().split())
        P.append(p)
        T.append(t)
    Q = int(sys.stdin.readline())
    queries = [int(sys.stdin.readline()) for _ in range(Q)]

    # Calculate the arrival time for each query
    arrival_times = calculate_arrival_time(N, X, Y, P, T, Q, queries)

    # Write the arrival times to stdout
    for time in arrival_times:
        sys.stdout.write(str(time) + "
")

if __name__ == "__main__":
    main()