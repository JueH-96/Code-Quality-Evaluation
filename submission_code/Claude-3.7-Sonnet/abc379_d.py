from collections import deque

Q = int(input())  # Number of queries

current_time = 0  # Total time passed
plants_by_birth = {}  # Dictionary to store the number of plants planted at each time
birth_times = deque()  # Deque of unique birth times, sorted by planting time

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Plant a new plant with height 0
        if current_time in plants_by_birth:
            plants_by_birth[current_time] += 1
        else:
            plants_by_birth[current_time] = 1
            birth_times.append(current_time)
    
    elif query[0] == '2':
        # Wait for T days
        T = int(query[1])
        current_time += T
    
    elif query[0] == '3':
        # Harvest plants with height at least H
        H = int(query[1])
        harvested = 0
        
        # Harvest plants with birth time <= current_time - H
        threshold = current_time - H
        while birth_times and birth_times[0] <= threshold:
            harvested += plants_by_birth[birth_times[0]]
            del plants_by_birth[birth_times[0]]
            birth_times.popleft()
        
        print(harvested)