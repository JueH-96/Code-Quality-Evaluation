Q = int(input())
current_time = 0
plants = []  # List of planting times

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Plant a new plant
        plants.append(current_time)
    elif query[0] == 2:
        # Wait T days
        T = query[1]
        current_time += T
    else:  # query[0] == 3
        # Harvest plants with height >= H
        H = query[1]
        harvested = 0
        remaining_plants = []
        
        for plant_time in plants:
            height = current_time - plant_time
            if height >= H:
                harvested += 1
            else:
                remaining_plants.append(plant_time)
        
        plants = remaining_plants
        print(harvested)