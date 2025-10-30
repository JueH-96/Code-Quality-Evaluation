def solve():
    n = int(input())
    slimes = []
    for _ in range(n):
        s, c = map(int, input().split())
        slimes.append((s, c))

    slimes.sort()
    
    total_slimes = 0
    
    counts = {}
    for s, c in slimes:
        counts[s] = c
        
    sizes = sorted(counts.keys())
    
    for size in sizes:
        count = counts[size]
        
        total_slimes += count % 2
        
        counts[size * 2] = counts.get(size * 2, 0) + count // 2
        
    
    total_slimes += sum(counts[size] for size in counts if counts[size] % 2 == 1)
    
    counts_even = {}
    for size in counts:
        counts_even[size] = counts[size] // 2 * 2
    
    
    total_slimes = 0
    
    counts = {}
    for s, c in slimes:
        counts[s] = c
        
    sizes = sorted(counts.keys())
    
    for size in sizes:
        count = counts[size]
        
        total_slimes += count % 2
        
        counts[size * 2] = counts.get(size * 2, 0) + count // 2
        
    
    total_slimes += sum(counts[size] for size in counts if counts[size] > 0)
    
    
    print(total_slimes)

solve()