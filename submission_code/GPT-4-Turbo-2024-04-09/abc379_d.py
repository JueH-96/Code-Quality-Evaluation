import sys
input = sys.stdin.read

def solve():
    data = input().split()
    Q = int(data[0])
    queries = data[1:]
    
    # This will hold the results for type 3 queries
    results = []
    
    # This will hold the base height of all plants
    global_height = 0
    
    # This will hold the actual heights of plants at the time they were planted
    plants = []
    
    index = 0
    while index < len(queries):
        query_type = int(queries[index])
        
        if query_type == 1:
            # Add a new plant with current global height as its base height
            plants.append(global_height)
            index += 1
        
        elif query_type == 2:
            # Increase global height by T
            T = int(queries[index + 1])
            global_height += T
            index += 2
        
        elif query_type == 3:
            # Harvest plants with height at least H
            H = int(queries[index + 1])
            effective_height = H - global_height
            count = 0
            # We will collect plants that are not harvested
            new_plants = []
            for plant_height in plants:
                if plant_height + global_height >= H:
                    count += 1
                else:
                    new_plants.append(plant_height)
            plants = new_plants
            results.append(str(count))
            index += 2
    
    # Print all results for type 3 queries
    print("
".join(results))