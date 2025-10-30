# YOUR CODE HERE
def process_queries(queries):
    plants = []
    results = []
    current_time = 0
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            # Add a new plant with height 0
            plants.append(current_time)
        elif parts[0] == '2':
            # Increase the current time by T
            T = int(parts[1])
            current_time += T
        elif parts[0] == '3':
            # Harvest plants with height >= H
            H = int(parts[1])
            # Calculate the actual height of each plant and count those >= H
            harvested_count = 0
            new_plants = []
            for plant_time in plants:
                if current_time - plant_time >= H:
                    harvested_count += 1
                else:
                    new_plants.append(plant_time)
            plants = new_plants
            results.append(harvested_count)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    Q = int(data[0])
    queries = data[1:Q+1]
    
    results = process_queries(queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()