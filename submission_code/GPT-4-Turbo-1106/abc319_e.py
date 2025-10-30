import math

# Read the number of bus stops, time to walk to the first bus stop, and time to walk from the last bus stop to Aoki's house
N, X, Y = map(int, input().split())

# Read the bus departure intervals and travel times for each bus stop
buses = [tuple(map(int, input().split())) for _ in range(N - 1)]

# Read the number of queries
Q = int(input())

# Read each query
queries = [int(input()) for _ in range(Q)]

# Process each query
for q in queries:
    # Start with the time Takahashi leaves his house
    current_time = q + X
    
    # Go through each bus stop
    for P, T in buses:
        # Wait for the next bus if necessary
        if current_time % P != 0:
            current_time += P - (current_time % P)
        # Travel to the next bus stop
        current_time += T
    
    # Walk from the last bus stop to Aoki's house
    current_time += Y
    
    # Print the earliest time Takahashi can arrive at Aoki's house
    print(current_time)