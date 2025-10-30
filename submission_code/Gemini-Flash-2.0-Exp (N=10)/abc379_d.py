def solve():
    q = int(input())
    plants = []
    total_growth = 0
    output = []

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            plants.append(0 - total_growth)
        elif query[0] == 2:
            total_growth += query[1]
        elif query[0] == 3:
            harvest_height = query[1]
            harvested_count = 0
            new_plants = []
            for height in plants:
                if height + total_growth >= harvest_height:
                    harvested_count += 1
                else:
                    new_plants.append(height)
            plants = new_plants
            output.append(harvested_count)
    
    for count in output:
        print(count)

solve()