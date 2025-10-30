def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]
    
    min_stands = float('inf')
    
    for i in range(1 << n):
        stands_selected = []
        for j in range(n):
            if (i >> j) & 1:
                stands_selected.append(j)
        
        if not stands_selected:
            continue
            
        flavors_covered = [False] * m
        for stand_index in stands_selected:
            for flavor_index in range(m):
                if s[stand_index][flavor_index] == 'o':
                    flavors_covered[flavor_index] = True
        
        all_flavors_covered = all(flavors_covered)
        
        if all_flavors_covered:
            min_stands = min(min_stands, len(stands_selected))
            
    print(min_stands)

solve()