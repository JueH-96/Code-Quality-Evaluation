Q = int(input())
plants = []
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        plants.append(0)
    elif query[0] == 2:
        T = query[1]
        for i in range(len(plants)):
            plants[i] += T
    else:
        H = query[1]
        count = 0
        new_plants = []
        for plant in plants:
            if plant < H:
                new_plants.append(plant)
            else:
                count += 1
        plants = new_plants
        print(count)