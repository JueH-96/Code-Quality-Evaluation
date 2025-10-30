def solve():
    q = int(input())
    queries = []
    for _ in range(q):
        line = list(map(int, input().split()))
        queries.append(line)
    
    plant_planting_times = []
    current_day = 0
    output_harvest_counts = []
    
    for query in queries:
        query_type = query[0]
        if query_type == 1:
            plant_planting_times.append(current_day)
        elif query_type == 2:
            t = query[1]
            current_day += t
        elif query_type == 3:
            h = query[1]
            harvest_threshold_day = current_day - h
            harvested_count = 0
            remaining_planting_times = []
            for planting_time in plant_planting_times:
                if planting_time <= harvest_threshold_day:
                    harvested_count += 1
                else:
                    remaining_planting_times.append(planting_time)
            plant_planting_times = remaining_planting_times
            output_harvest_counts.append(harvested_count)
            
    for count in output_harvest_counts:
        print(count)

if __name__ == '__main__':
    solve()