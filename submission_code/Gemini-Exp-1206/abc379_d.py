def solve():
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
        else:
            h = query[1]
            harvested_count = 0
            new_plants = []
            for plant_height in plants:
                if plant_height >= h:
                    harvested_count += 1
                else:
                    new_plants.append(plant_height)
            plants = new_plants
            print(harvested_count)

solve()