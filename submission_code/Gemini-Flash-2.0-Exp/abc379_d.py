Q = int(input())
plants = []
total_growth = 0
output = []

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        plants.append(0 - total_growth)
    elif query[0] == 2:
        T = query[1]
        total_growth += T
    else:
        H = query[1]
        harvested = 0
        new_plants = []
        for plant in plants:
            if plant + total_growth >= H:
                harvested += 1
            else:
                new_plants.append(plant)
        plants = new_plants
        output.append(harvested)

for res in output:
    print(res)