# YOUR CODE HERE
from collections import defaultdict

def main():
    Q = int(input())
    plants = defaultdict(int)
    total_plants = 0
    current_time = 0

    for _ in range(Q):
        query = input().split()
        
        if query[0] == '1':
            plants[current_time] += 1
            total_plants += 1
        
        elif query[0] == '2':
            current_time += int(query[1])
        
        else:  # query[0] == '3'
            harvest_height = int(query[1])
            harvested = 0
            new_plants = defaultdict(int)
            
            for plant_time, count in plants.items():
                plant_height = current_time - plant_time
                if plant_height >= harvest_height:
                    harvested += count
                    total_plants -= count
                else:
                    new_plants[plant_time] = count
            
            plants = new_plants
            print(harvested)

if __name__ == "__main__":
    main()