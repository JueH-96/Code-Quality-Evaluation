def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    Q = int(data[0])
    plants = []
    height_increase = 0
    output = []
    
    for i in range(1, Q + 1):
        query = data[i].split()
        if query[0] == '1':
            # Prepare one empty flower pot and put a plant in it (height 0)
            plants.append(0)
        elif query[0] == '2':
            # Wait for T days, increase height of all plants
            T = int(query[1])
            height_increase += T
        elif query[0] == '3':
            # Harvest all plants with height at least H
            H = int(query[1])
            count_harvested = 0
            
            # Calculate the effective height of each plant
            new_plants = []
            for plant in plants:
                effective_height = plant + height_increase
                if effective_height >= H:
                    count_harvested += 1
                else:
                    new_plants.append(plant)
            
            # Update the plants list to only include those that were not harvested
            plants = new_plants
            output.append(str(count_harvested))
    
    # Print all results for type 3 queries
    print("
".join(output))

if __name__ == "__main__":
    main()