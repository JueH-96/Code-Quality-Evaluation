q = int(input())
plants = []
for _ in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        plants.append(0)
    elif query[0] == 2:
        t = query[1]
        for i in range(len(plants)):
            plants[i] += t
    elif query[0] == 3:
        h = query[1]
        count = 0
        new_plants = []
        for plant in plants:
            if plant >= h:
                count += 1
            else:
                new_plants.append(plant)
        plants = new_plants
        print(count)