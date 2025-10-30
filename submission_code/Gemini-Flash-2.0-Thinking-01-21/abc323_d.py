def solve():
    n = int(input())
    initial_slimes = []
    for _ in range(n):
        s_c = list(map(int, input().split()))
        initial_slimes.append({'size': s_c[0], 'count': s_c[1]})
    
    counts = {}
    for slime in initial_slimes:
        counts[slime['size']] = slime['count']
        
    while True:
        sizes = sorted(list(counts.keys()))
        made_synthesis = False
        next_counts = {}
        sizes_processed_in_this_iteration = set()
        
        for size in sizes:
            if size in sizes_processed_in_this_iteration:
                continue
            sizes_processed_in_this_iteration.add(size)
            if counts.get(size, 0) >= 2:
                current_count = counts[size]
                syntheses = current_count // 2
                remaining_count = current_count % 2
                counts[size] = remaining_count
                if syntheses > 0:
                    next_size = 2 * size
                    counts[next_size] = counts.get(next_size, 0) + syntheses
                    made_synthesis = True
                    
        if not made_synthesis:
            break
            
    total_slimes = 0
    for size in counts:
        total_slimes += counts[size]
        
    print(total_slimes)

if __name__ == '__main__':
    solve()