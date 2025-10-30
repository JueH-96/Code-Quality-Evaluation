# YOUR CODE HERE
Q = int(input())
plants = []  # List of birth times
current_time = 0

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Add a new plant with birth time = current_time
        plants.append(current_time)
    
    elif query[0] == 2:
        # Wait T days
        T = query[1]
        current_time += T
    
    elif query[0] == 3:
        # Harvest plants with height >= H
        H = query[1]
        harvested = 0
        remaining_plants = []
        
        for birth_time in plants:
            height = current_time - birth_time
            if height >= H:
                harvested += 1
            else:
                remaining_plants.append(birth_time)
        
        plants = remaining_plants
        print(harvested)