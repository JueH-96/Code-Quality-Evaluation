# YOUR CODE HERE
import sys
input = sys.stdin.read

def process_queries(queries):
    plants = []
    current_time = 0
    results = []
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            plants.append(current_time)
        elif parts[0] == '2':
            T = int(parts[1])
            current_time += T
        elif parts[0] == '3':
            H = int(parts[1])
            count = 0
            new_plants = []
            for plant_time in plants:
                if current_time - plant_time >= H:
                    count += 1
                else:
                    new_plants.append(plant_time)
            plants = new_plants
            results.append(count)
    
    return results

def main():
    data = input().splitlines()
    Q = int(data[0])
    queries = data[1:Q+1]
    results = process_queries(queries)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()