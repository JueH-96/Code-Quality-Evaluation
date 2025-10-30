def solve():
    n = int(input())
    initial_slimes = []
    for _ in range(n):
        s_c = list(map(int, input().split()))
        initial_slimes.append({'size': s_c[0], 'count': s_c[1]})
    
    counts = {}
    for slime in initial_slimes:
        counts[slime['size']] = slime['count']
        
    sorted_initial_sizes = sorted([slime['size'] for slime in initial_slimes])
    
    for size in sorted_initial_sizes:
        current_size = size
        while True:
            count = counts.get(current_size, 0)
            if count < 2:
                break
            syntheses = count // 2
            counts[current_size] = count % 2
            next_size = 2 * current_size
            counts[next_size] = counts.get(next_size, 0) + syntheses
            current_size = next_size
            
    total_slimes = 0
    for size in counts:
        total_slimes += counts[size]
        
    print(total_slimes)

if __name__ == '__main__':
    solve()