def solve():
    n = int(input())
    x_coords = list(map(int, input().split()))
    populations = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append({'l': l, 'r': r})
    
    prefix_population_sum = [0] * n
    prefix_population_sum[0] = populations[0]
    for i in range(1, n):
        prefix_population_sum[i] = prefix_population_sum[i-1] + populations[i]
        
    results = []
    for query in queries:
        l_range = query['l']
        r_range = query['r']
        start_index = -1
        end_index = -1
        
        # Find start_index (first index where x_coords[i] >= l_range)
        low = 0
        high = n
        start_village_index = n
        while low < high:
            mid = (low + high) // 2
            if x_coords[mid] >= l_range:
                start_village_index = mid
                high = mid
            else:
                low = mid + 1
        
        # Find end_index (last index where x_coords[i] <= r_range)
        low = -1
        high = n - 1
        end_village_index = -1
        while low < high:
            mid = (low + high + 1) // 2
            if x_coords[mid] <= r_range:
                end_village_index = mid
                low = mid
            else:
                high = mid - 1
                
        if start_village_index > end_village_index:
            total_villagers = 0
        else:
            start_idx = start_village_index
            end_idx = end_village_index
            if start_idx > end_idx:
                total_villagers = 0
            else:
                if start_idx == 0:
                    total_villagers = prefix_population_sum[end_idx]
                else:
                    total_villagers = prefix_population_sum[end_idx] - prefix_population_sum[start_idx-1]
                    
        results.append(total_villagers)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()