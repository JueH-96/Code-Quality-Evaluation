def solve():
    n = int(input())
    x_coords = list(map(int, input().split()))
    populations = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append({'l': l, 'r': r})
    
    prefix_population_sum = [0] * (n + 1)
    for i in range(n):
        prefix_population_sum[i+1] = prefix_population_sum[i] + populations[i]
        
    results = []
    for query in queries:
        l_range = query['l']
        r_range = query['r']
        start_index = -1
        low = 0
        high = n - 1
        while low <= high:
            mid = (low + high) // 2
            if x_coords[mid] >= l_range:
                start_index = mid
                high = mid - 1
            else:
                low = mid + 1
                
        end_index = -1
        low = 0
        high = n - 1
        while low <= high:
            mid = (low + high) // 2
            if x_coords[mid] <= r_range:
                end_index = mid
                low = mid + 1
            else:
                high = mid - 1
                
        if start_index == -1 or end_index == -1 or start_index > end_index:
            results.append(0)
        else:
            population_sum = prefix_population_sum[end_index + 1] - prefix_population_sum[start_index]
            results.append(population_sum)
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()