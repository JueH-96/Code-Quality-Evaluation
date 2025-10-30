def solve():
    n = int(input())
    beans = []
    for _ in range(n):
        a, c = map(int, input().split())
        beans.append((a, c))
    
    color_map = {}
    for a, c in beans:
        if c not in color_map:
            color_map[c] = []
        color_map[c].append(a)
    
    max_min_deliciousness = 0
    for color, deliciousness_list in color_map.items():
        min_deliciousness = min(deliciousness_list)
        max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)
        
    print(max_min_deliciousness)

solve()